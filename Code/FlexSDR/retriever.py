import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class BilinearScore(nn.Module):
    def __init__(self, query_dim, demon_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.demon_proj = nn.Linear(demon_dim, hidden_dim, bias=False)
    
    def forward(self, query_embed, demon_embed):
        query_embed = self.query_proj(query_embed)
        demon_embed = self.demon_proj(demon_embed)
        score = torch.matmul(query_embed, demon_embed.T)
        return score


class SequenceEncode(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, model_name='lstm'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_name = model_name
        # think about project embedding to hidden dim
        self.qk_proj = nn.Linear(input_dim, int(hidden_dim / 2))
        self.qq_proj = nn.Linear(input_dim, int(hidden_dim / 2))
        self.dq_proj = nn.Linear(input_dim, hidden_dim)
        self.eos_embed = nn.Parameter(torch.randn(hidden_dim)[None,:]) # <EOS> token [1,D]
        self.bos_embed = nn.Parameter(torch.randn(hidden_dim)[None,:]) # <BOS> token [1,D]

        if self.model_name == 'lstm':
            self.rnn_layer = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                     batch_first=True, dropout=dropout)
        else:
            self.rnn_layer = nn.GRU(hidden_dim, hidden_dim, num_layers, 
                                    batch_first=True, dropout=dropout)
            

    def forward(self, qk_input, qq_input, dq_input, hist_select):
        """This function accept sequence inputs and generate next step result.
        The output of this function is the estimated value for the status and the action scores

        qk_input: query knowledge input [B,D]
        qq_input: query question input [B,D] (h0 for sequence)
        dq_input: demonstration question input [M,D]
        hist_select: padded history selection records [B,L,M+1], 0 is <eos>, (1 is first demonstration)
        qd_mask: same query demonstration mask (global) [B,M]
        """
        qk_embed = self.qk_proj(qk_input) # [B,D]
        qq_embed = self.qq_proj(qq_input) # [B,D]
        
        dq_embed = self.dq_proj(dq_input) # [M,D]
        # add the stop token input the candidation demonstration
        dq_embed = torch.cat([self.eos_embed, dq_embed], dim=0) # [M+1,D]

        # initialized h0 as query question and knowledge concatenation
        h0 = torch.cat([qk_embed, qq_embed], dim=1)[None,:,:].repeat(self.num_layers,1,1) # [L,B,D]
        c0 = torch.zeros_like(h0)

        # based on the selectoin, we can always like add an inital <BOS> token for simplicity
        hist_embed = torch.matmul(hist_select.float(), dq_embed[None,:,:].repeat(hist_select.shape[0],1,1)) #[B,L,D]
        bos_embed = self.bos_embed[None,:,:].repeat(hist_embed.shape[0],1,1) # [B,1,D]
        hist_input = torch.cat([bos_embed, hist_embed], dim=1) #[B,L+1,D]
        if self.model_name == 'lstm':
            _, (ht, _) = self.rnn_layer(hist_input, (h0, c0))
        else:
            _, ht = self.rnn_layer(hist_input, h0)
        return ht, dq_embed


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, early_stop=True, model_name='lstm', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # encoder layer: encode the sequential status into status vector
        self.encode_layer = SequenceEncode(input_dim, hidden_dim, num_layers, dropout, model_name)

        # critic layer: generate the value estimation
        self.critic_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # action layer: accept demonstration and query, generate evaluation scores
        self.actor_layer = BilinearScore(hidden_dim, hidden_dim, 64)
        self.early_stop = early_stop


    def forward(self, qk_input, qq_input, dq_input, hist_select, qd_mask=None):
        """This function accept sequence inputs and generate next step result.
        The output of this function is the estimated value for the status and the action scores

        qk_input: query knowledge input [B,D]
        qq_input: query question input [B,D] (h0 for sequence)
        dq_input: demonstration question input [M,D]
        hist_select: padded history selection records [B,L,M+1], 0 is <eos>, (1 is first demonstration)
        hist_length: history length information [B,]
        qd_mask: same query demonstration mask (global) [B,M]
        """        
        ht, dq_embed = self.encode_layer(qk_input, qq_input, dq_input, hist_select)
        # find out the last layer hidden status
        value = self.critic_layer(ht[-1]) # [B,]
        score = self.actor_layer(ht[-1], dq_embed) # [B,M+1]
        # based on prior select result, we need to mask partial logits
        score_mask = (hist_select.sum(dim=1) > 0).float() # [B,M+1]
        score_mask_0 = (score_mask[:,0]==0).float() # [B,] pick out the ones without selecting <EOS> in history
        score_mask_1 = torch.Tensor([0,]+[1,]*(dq_embed.shape[0]-1)).to(dq_embed.device) # [M+1,]
        score_mask_1 = score_mask_1[None,:].repeat(score_mask.shape[0],1) # [B,M+1]
        score_mask = 1 - (score_mask * score_mask_0[:,None] + score_mask_1 * (1-score_mask_0)[:,None]) # [B,M+1]
        # apart from the history based mask, we also need to mask the unrelated query's demonstrations
        if qd_mask is not None:
            score_mask_2 = torch.cat([torch.ones(qd_mask.shape[0],1).to(qd_mask.device),qd_mask], dim=1) # [B,M+1]
            score_mask = score_mask * score_mask_2

        if not self.early_stop:
            # ensure the first column will not be selected
            score_mask[:,0] = 0
            
        score = score * score_mask - 1e9 * (1 - score_mask)
        score = F.softmax(score, dim=1) # [B,M+1]

        return score, value[:,0]


    def generate(self, qk_input, qq_input, dq_input, qd_mask=None, max_length=10, do_sample=True):
        """This function accept query question and knowledge, generate demonstration index directly
        qk_input: query knowledge input [B,D]
        qq_input: query question input [B D]
        dq_input: demonstration question input [M,D]
        qd_mask: same query demonstration mask [B,M]
        max_length: max allowed length for demonstrations
        do_sample: if True doing sampling as generation, if False doing greedy generation
        """
        qk_embed = self.encode_layer.qk_proj(qk_input) # [B,D]
        qq_embed = self.encode_layer.qq_proj(qq_input) # [B,D]
        
        dq_embed = self.encode_layer.dq_proj(dq_input) # [M,D]
        dq_embed = torch.cat([self.encode_layer.eos_embed, dq_embed], dim=0) # [M+1,D]

        # input_t = torch.cat([qk_embed, qq_embed], dim=1) # [B, D]
        input_t = self.encode_layer.bos_embed.repeat(qk_embed.shape[0],1) #[B, D]
        
        ht = torch.cat([qk_embed, qq_embed], dim=1)[None,:,:].repeat(self.encode_layer.num_layers,1,1) # [L,B,D]
        ct = torch.zeros_like(ht)
        
        # this is used for calculate valid score logits during the inference
        helper_masks = torch.diag(torch.Tensor([0,] + [1,] * (dq_embed.shape[0] - 1)))
        helper_masks[0] = torch.Tensor([0,] + [1,] * (dq_embed.shape[0] - 1))
        helper_masks = helper_masks.to(dq_embed.device)
        # cache the selected index, for output as the result
        select_cache, logits_cache, values_cache, valids_cache = [], [], [], []
        # cache the selected index, for ruling out the duplication in selction process
        # logits_masks = torch.zeros([input_t.shape[0], dq_embed.shape[0]]).to(input_t.device) # [B, M+1]

        if qd_mask is None:
            logits_masks = torch.zeros([input_t.shape[0], dq_embed.shape[0]]).to(input_t.device) # [B, M+1]
        else:
            logits_masks = 1 - torch.cat([torch.ones(qd_mask.shape[0],1).to(qd_mask.device),qd_mask],dim=1)

        if not self.early_stop:
            # ensure the first column will not be selected (risky!!!)
            logits_masks[:,0] = 1
        
        # cache the logits for later calculate the unbiased rewards
        for t in range(max_length):

            if self.encode_layer.model_name == 'lstm':
                _, (ht, ct) = self.encode_layer.rnn_layer(input_t[:,None,:], (ht, ct))
            else:
                _, ht = self.encode_layer.rnn_layer(input_t[:,None,:], ht)

            scores_t = self.actor_layer(ht[-1], dq_embed)

            # the softmax mask also need to have the very negative value to ensure its work
            logits_t = F.softmax(scores_t * (1 - logits_masks) - 1e9 * logits_masks, dim=1) 

            # calculate the value for each step used for later reward with baseline calculation
            value_t = self.critic_layer(ht) # [B,1]
            values_cache.append(value_t)

            if do_sample:
                # use the sampleing method
                dist = Categorical(logits_t)
                select_t = dist.sample() # [B,]
                select_logp = dist.log_prob(select_t)
            else:
                # use the greedy method
                select_p, select_t = torch.max(logits_t, dim=1)
                select_logp = torch.log(select_p)
            
            select_t = F.one_hot(select_t, num_classes=dq_embed.shape[0]).to(dq_embed.device) # [B, M+1]
            # update the input_t for next step, simplely by selecting the corresponding embed from demonstration
            input_t = torch.matmul(select_t[:,None,:].float(), dq_embed[None,:,:].repeat(input_t.shape[0], 1, 1)).squeeze(1) # [B, D]
            # update the logtics cache for later reward calculation
            select_cache.append(select_t) 
            logits_cache.append(select_logp)
            # update valids mask for indicate some invalid outputs for logtis calcuation on 1 result
            valids_cache.append(((logits_masks - helper_masks[0][None,:]).abs().sum(dim=1) == 0).float()) # [B,]
            # based on the select t we may need to calculate the mask matrix for the remaining step
            logits_masks = ((logits_masks + torch.matmul(select_t.float(), helper_masks)) > 0).float()
            
        # after running, we return the selected idex and also calculate its probability value
        select_cache = torch.stack(select_cache, dim=1) # [B, T, M+1]
        valids_cache = torch.stack(valids_cache, dim=1) # [B, T]
        logits_cache = torch.stack(logits_cache, dim=1) # [B, T]
        values_cache = torch.stack(values_cache, dim=1) # [B, T]
        
        return select_cache, logits_cache, values_cache, 1 - valids_cache

    def evaluate(self, qk_input, qq_input, dq_input, action_sequence, qd_mask):
        """generate the evaluation results for the given action sequence and query information
        action_sequence: [B,T,M+1]
        """
        action_logprobs, state_values, dist_entropy = [], [], []
        for i in range(action_sequence.shape[1]):
            hist_select = action_sequence[:,:i,:]
            score, value = self.forward(qk_input, qq_input, dq_input, hist_select, qd_mask)
            state_values.append(value)
            # based on these, we can calculate the output results
            # pick out the log score based on the action score -> [B,M+1] value -> [B,]
            dist = Categorical(score)
            logprob = dist.log_prob(action_sequence[:,i,:].argmax(dim=1)) # [B,]
            action_logprobs.append(logprob)
            entropy = dist.entropy() # [B,]
            dist_entropy.append(entropy)
        
        action_logprobs = torch.stack(action_logprobs, dim=1) #[B, T]
        state_values = torch.stack(state_values, dim=1) #[B, T]
        dist_entropy = torch.stack(dist_entropy, dim=1) #[B, T]

        return action_logprobs, state_values, dist_entropy