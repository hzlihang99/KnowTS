import os
import re
import sys
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.generator import RemoteGenerator, LocalGenerator
from Tools.utils import textlist_openaiembed

PROCESS_NUM=40
EMBED_MODEL_NAME='text-embedding-3-small'
EMBED_BATCH_SIZE=256

device = 'cuda'

def prepare_zerohot(request_text, style='cot'):
    system_text = ['Your are a helpful assistant.',
                   'Please provide correct solution to the given math word problem.',
                   'The solution value should be provided at the end of the reponse, and in the format like #### X, where X is solution value.']
    system_text = ' '.join(system_text)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": request_text},
    ]
    return messages


class Dataset(object):
    def __init__(self, data_list, question_embed):
        self.data_list = data_list
        self.question_embed = question_embed
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        sample['question_embed'] = torch.FloatTensor(self.question_embed[index])
        return sample


class Retriver(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.project_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, query_x, demon_x):
        query_x = self.project_layer(query_x)
        demon_x = self.project_layer(demon_x)
        # calculate the score
        scores = torch.matmul(query_x, demon_x.T)
        return scores


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--config_path', type=str, required=True, help='configuration path.')
    parser.add_argument("--do_train", action='store_true', help='do training on train set.')
    parser.add_argument("--do_eval", action='store_true', help='do evaluation on test set.')
    parser.add_argument("--local_cache", action='store_true', help='search and insert cache in local machine.')
    parser.add_argument("--search_cache", action='store_true', help='search cache for LLM inference result.')
    parser.add_argument("--insert_cache", action='store_true', help='insert the LLM result into cache.')
    parser.add_argument("--run_id", type=str, default=None, help='runid, used for multiple run and average result')
    args = parser.parse_args()

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    
    # ---- Training Setting ---- #
    epoch_num = config['train']['n_epochs']
    learning_rate = config['train']['learning_rate']

    local_cache = args.local_cache
    search_cache = args.search_cache
    insert_cache = args.insert_cache

    # ---- Parameter for LLM (Enviroment) ---- #
    llm_model = config['llm_agent']['model_name']

    # ---- Genearl and Data Setting ---- #
    label_path = config['data']['label_path']
    fewshot_maxsize = config['data']['fewshot_maxsize']
    back_every_steps = config['general']['back_every_steps']

    experiment_name = config['general']['experiment_name']
    if args.run_id is not None:
        experiment_name = experiment_name + '-rid-{}'.format(args.run_id)
    
    save_root = "../../Save/{}".format(experiment_name)
    os.makedirs(save_root, exist_ok=True)
    shutil.copy2(args.config_path, save_root)

    # ---- Build up the Datasets ---- #

    label_data = pd.read_csv(label_path, sep='\t')

    question_embed_path = os.path.join(save_root, 'question_embed_cache.npy')
    if os.path.exists(question_embed_path):
        print('find cached question embed, using it for experiment...')
        question_embed = np.load(question_embed_path)
    else:
        print('cannot find cached question embed, run embedding api...')
        question_embed = textlist_openaiembed(label_data['question'].tolist(), EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
        print('save cached question embed to {}'.format(question_embed_path))
        np.save(question_embed_path, question_embed)

    print('Preapre train set...')
    train_flag = label_data['demon_flag'] == -1
    train_data = label_data[train_flag].to_dict('records')
    train_data = Dataset(train_data, question_embed[train_flag.values])
    train_loader = DataLoader(train_data, config['train']['batch_size'], shuffle=True)
    
    print('Preapre valid set...')
    valid_flag = label_data['demon_flag'] == 0
    valid_data = label_data[valid_flag].to_dict('records')
    valid_data = Dataset(valid_data, question_embed[valid_flag.values])
    valid_loader = DataLoader(valid_data, config['valid']['batch_size'], shuffle=False)

    print('Preapre demondata...')
    demon_flag = label_data['demon_flag']==1
    demon_data = label_data[demon_flag].reset_index(drop=True)
    demon_question_embed = torch.FloatTensor(question_embed[demon_flag.values]).to(device)

    # ---- Initilaize the PromptPG and LLM agent ---- #

    model = Retriver(config['retriver']['input_dim'], 
                     config['retriver']['hidden_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if 'gpt' in llm_model:
        llm_agent = RemoteGenerator(llm_model, PROCESS_NUM, search_cache, insert_cache, local_cache)
    else:
        llm_agent = LocalGenerator(llm_model, config['llm_agent']['batch_size'], PROCESS_NUM, 
                                   search_cache=search_cache, insert_cache=insert_cache, local_cache=local_cache)
    
    if args.do_train:
        for epoch in range(epoch_num):
            model.train()    
            batch_progress = tqdm(train_loader, desc='Epoch {:0>2d}'.format(epoch))
            for batch_data in batch_progress:
                query_question_input = batch_data['question_embed'].to(device)
                scores = model(query_question_input, demon_question_embed)
                scores = F.softmax(scores, dim=1)

                batch_mask, messages_list = [], []
                for i in range(scores.shape[0]):
                    query_question = batch_data['question'][i]
                    request_text = query_question
                    # prepare for the inital message prompt
                    messages = prepare_zerohot(request_text, 'cot')

                    # prepare for the interaction
                    demon_prob = scores[i,:].clone().detach() 
                    demon_prob = demon_prob.cpu().numpy()
                    demon_prob = np.nan_to_num(demon_prob, nan=0.000001)  # replace np.nan with 0
                    demon_prob /= demon_prob.sum()  # make probabilities sum to 1

                    demon_index = np.random.choice(range(len(demon_data)), fewshot_maxsize, p=demon_prob, replace=False)
                    demon_index = demon_index[::-1].copy()

                    demon_mask = torch.zeros_like(scores[i,:])
                    demon_mask[demon_index] = 1
                    batch_mask.append(demon_mask)

                    for j in demon_index:
                        demon_question = demon_data['question'].iloc[j]
                        demon_response = demon_data['answer'].iloc[j]
                        request_text = demon_question
                        messages.insert(-1, {"role": "user", "content":request_text})
                        messages.insert(-1, {"role": "assistant", "content":demon_response})

                    if 'mistralai' in llm_model:
                        messages_mistra = [{"role":"user", "content":'\n'.join([x['content'] for x in messages[:2]])}] + messages[2:]
                        messages_list.append((i, messages_mistra))
                    else:
                        messages_list.append((i, messages))

                batch_mask = torch.stack(batch_mask, dim=0)
                batch_logprob = (torch.log(scores + 1e-9) * batch_mask).sum(dim=1)

                batch_response = llm_agent.generate(messages_list)
                batch_response = [y[1] for y in sorted(batch_response, key=lambda x: x[0])]

                model_value = []
                for x in batch_response:
                    try:
                        model_value.append(eval(re.findall(r'####? ?\$?([\+\-]?\d+\.?\d*)', x)[0]))
                    except:
                        model_value.append(None)
                human_value = [eval(re.findall(r'#### ?([\+\-]?\d+)', x)[0]) for x in batch_data['answer']]
                batch_reward = torch.Tensor([x==y for x,y in zip(model_value, human_value)]).to(device)
                batch_reward = batch_reward.float() * 2 - 1
                batch_loss = -1 * batch_reward * batch_logprob
                batch_loss = batch_loss.mean()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                batch_progress.set_description("Epoch {:0>2d} - Loss: {:.4f} - Reward: {:.4f}".format(
                    epoch, batch_loss.cpu().detach().numpy(), batch_reward.mean().cpu().detach().numpy()
                ))

            if epoch % back_every_steps == 0:
                print('save the model...')
                save_path = os.path.join(save_root, 'epoch_{}.pt'.format(epoch))
                torch.save(model.state_dict(), save_path)

    if args.do_eval:
        for save_name in os.listdir(save_root):
            if save_name[-3:] != '.pt':
                continue
            print('='*20, ' ', save_name[:-3], ' ', '='*20) 
            save_path = os.path.join(save_root, save_name)
            save_model = torch.load(save_path, map_location=device)
            model.load_state_dict(save_model)
            model.eval()

            with torch.no_grad():
                model_values, human_values = [], []
                for batch_data in tqdm(valid_loader):
                    query_question_embed = batch_data['question_embed'].to(device)
                    scores = model(query_question_embed, demon_question_embed)
                    scores = F.softmax(scores, dim=1)

                    messages_list = []
                    for i in range(scores.shape[0]):
                        query_question = batch_data['question'][i]
                        request_text = query_question
                        # prepare for the inital message prompt
                        messages = prepare_zerohot(request_text, 'cot')

                        demon_prob = scores[i,:].clone().detach() 
                        demon_prob = demon_prob.cpu().numpy()
                        demon_prob = np.nan_to_num(demon_prob, nan=0.000001)  # replace np.nan with 0
                        demon_prob /= demon_prob.sum()  # make probabilities sum to 1

                        demon_index = np.random.choice(range(len(demon_data)), fewshot_maxsize, p=demon_prob, replace=False)
                        demon_index = demon_index[::-1].copy()

                        for j in demon_index:
                            demon_question = demon_data['question'].iloc[j]
                            demon_response = demon_data['answer'].iloc[j]
                            request_text = demon_question
                            messages.insert(-1, {"role": "user", "content":request_text})
                            messages.insert(-1, {"role": "assistant", "content":demon_response})

                        if 'mistralai' in llm_model:
                            messages_mistra = [{"role":"user", "content":'\n'.join([x['content'] for x in messages[:2]])}] + messages[2:]
                            messages_list.append((i, messages_mistra))
                        else:
                            messages_list.append((i, messages))

                    batch_response = llm_agent.generate(messages_list)
                    batch_response = [y[1] for y in sorted(batch_response, key=lambda x: x[0])]

                    model_value = []
                    for x in batch_response:
                        try:
                            model_value.append(eval(re.findall(r'####? ?\$?([\+\-]?\d+\.?\d*)', x)[0]))
                        except:
                            model_value.append(None)
                    human_value = [eval(re.findall(r'#### ?([\+\-]?\d+)', x)[0]) for x in batch_data['answer']]


                    model_values.append(np.array(model_value))
                    human_values.append(np.array(human_value))
                    
                model_values = np.concatenate(model_values)
                human_values = np.concatenate(human_values)

                print('='*20,os.path.basename(llm_model),'='*20)
                acc = (model_values == human_values).mean()
                print('Accuracy: {:.4f}'.format(acc))