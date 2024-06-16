import torch
import torch.nn as nn
from tqdm import tqdm
from retriever import ActorCritic

class Memory:
    def __init__(self):
        self.actions = None  # selected indicies
        self.logprobs = None # selection probability
        self.rewards = None  # sequence reward (run the program one-by-one)
        self.state_values = None # calculated state values for critic training
        self.action_valid = None # this variable is not sure to use as we may want to set it 
        
        self.qk_input = None
        self.qq_input = None
        self.dq_input = None
        self.qd_mask = None
    
    def clear(self):
        self.__init__() # clear the batch


class PPO:
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, 
                 lr_encode, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 early_stop=True, model_name='lstm', device='cuda'):

        self.device = device

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.model_name = model_name
        
        self.memory = Memory()

        self.policy = ActorCritic(input_dim, hidden_dim, num_layers, dropout, early_stop, model_name).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.encode_layer.parameters(), 'lr': lr_encode},
            {'params': self.policy.actor_layer.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_layer.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(input_dim, hidden_dim, num_layers, dropout, early_stop, model_name).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, qk_input, qq_input, dq_input, qd_mask, max_length=10):
        # add results into memory for later interaction with environment
        with torch.no_grad():
            action, action_logprob, state_value, action_valid = self.policy_old.generate(qk_input, qq_input, dq_input, qd_mask, max_length)
        
        self.memory.actions = action
        self.memory.logprobs = action_logprob
        self.memory.state_values = state_value
        self.memory.action_valid = action_valid
        self.memory.qd_mask = qd_mask

        self.memory.qk_input = qk_input
        self.memory.qq_input = qq_input
        self.memory.dq_input = dq_input
        self.memory.qd_mask  = qd_mask

        return action # the output shape is [B,T,M+1] and used for interaction with LLM

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []

        # no need to consider the zero term as it does not influence the calculated result 
        for reward_sequence in self.memory.rewards:
            discounted_reward = 0
            reward_cache = []
            for reward in reversed(reward_sequence):
                discounted_reward = reward + (self.gamma * discounted_reward)        
                reward_cache.insert(0, discounted_reward)
            rewards.append(reward_cache)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-7)

        # convert list to tensor
        old_actions = self.memory.actions.detach().to(self.device) # [B,T,M+1]
        old_actions_valid = self.memory.action_valid.detach().to(self.device)
        
        old_logprobs = self.memory.logprobs.detach().to(self.device) # [B,T]
        # old_logprobs = old_logprobs * old_actions_valid
        
        old_state_values = self.memory.state_values.detach().to(self.device) # [B,T]
        # old_state_values = old_state_values * old_actions_valid
        
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach() # [B,T]

        # Optimize policy for K epochs
        progress = tqdm(range(self.K_epochs))
        for _ in progress:
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                self.memory.qk_input, self.memory.qq_input, self.memory.dq_input, 
                old_actions, self.memory.qd_mask
            )
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()) # [B, T]

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # as the distribution of different reward paths are quite different
            # we may need to balance that part to ensure the model actually learn the things we want            

            loss = loss * old_actions_valid
            loss = loss.sum() / old_actions_valid.sum()

            # loss = loss.mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress.set_description("Loss: {:.4f}".format(loss.item()))
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.memory.clear()

    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))