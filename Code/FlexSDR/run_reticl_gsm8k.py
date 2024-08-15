import re
import os
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ppo_gsm8k import PPO
from generator import LocalGenerator, RemoteGenerator
from utils import textlist_openaiembed

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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--config_path', type=str, required=True, help='configuration path.')
    parser.add_argument("--do_train", action='store_true', help='do training on train set.')
    parser.add_argument("--do_eval", action='store_true', help='do evaluation on test set.')
    parser.add_argument("--local_cache", action='store_true', help='search and insert cache in local machine.')
    parser.add_argument("--search_cache", action='store_true', help='search cache for LLM inference result.')
    parser.add_argument("--insert_cache", action='store_true', help='write the LLM result into cache.')
    parser.add_argument("--run_id", type=str, default=None, help='runid, used for multiple run and average result')
    args = parser.parse_args()

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    # ---- Training Setting ---- #
    
    epoch_num = config['train']['n_epochs']
    lr_actor  = config['train']['lr_actor']
    lr_critic = config['train']['lr_critic']
    lr_encode = config['train']['lr_encode']

    local_cache = args.local_cache
    search_cache = args.search_cache
    insert_cache = args.insert_cache

    # ---- Parameter for PPO (Agent) ---- #
    
    gamma = config['ppo_agent']['gamma']
    eps_clip = config['ppo_agent']['eps_clip']
    K_epochs = config['ppo_agent']['K_epochs']

    input_dim = config['ppo_agent']['input_dim']
    hidden_dim = config['ppo_agent']['hidden_dim']
    n_layers = config['ppo_agent']['n_layers']
    dropout = config['ppo_agent']['dropout']
    ppo_model = config['ppo_agent']['model_name']

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
        print('Find cached question embed, using it for experiment...')
        question_embed = np.load(question_embed_path)
    else:
        print('Cannot find cached question embed, run embedding api...')
        question_embed = textlist_openaiembed(label_data['question'].tolist(), EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
        print('Cache question embed to {}'.format(question_embed_path))
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

    # ---- Initilaize the PPO and LLM agent ---- #
    
    ppo_agent = PPO(input_dim, hidden_dim, n_layers, dropout, 
                    lr_encode, lr_actor, lr_critic, gamma, 
                    K_epochs, eps_clip, False, ppo_model, device)

    if 'gpt' in llm_model:
        llm_agent = RemoteGenerator(llm_model, PROCESS_NUM, search_cache, insert_cache, local_cache)
    else:
        llm_agent = LocalGenerator(llm_model, config['llm_agent']['batch_size'], PROCESS_NUM, 
                                   search_cache=search_cache, insert_cache=insert_cache, local_cache=local_cache)

    if args.do_train:
        
        for epoch in range(epoch_num):
            
            batch_progress = tqdm(train_loader, desc='Epoch {:0>2d}'.format(epoch))
            
            for batch_data in batch_progress:
                
                # apply this for avoiding select the demonstration from other knowledges
                query_demon_mask = torch.ones(len(batch_data['question']), len(demon_data)).to(device)
                query_question_input = batch_data['question_embed'].to(device)
            
                print('start to generate action...')
                actions = ppo_agent.select_action(query_question_input, demon_question_embed, query_demon_mask, fewshot_maxsize)
                batch_index = actions.argmax(dim=2) - 1 # stop action is 0,  realign with the index in demonstration list
                
                batch_response = []
                print('start to interact with LLM...')
                messages_list = []
                for i in range(batch_index.shape[0]):
                    query_question = batch_data['question'][i]
                    request_text = query_question
                    messages = prepare_zerohot(request_text, 'cot')
                    
                    demon_index = batch_index[i].clone().detach().cpu().numpy()
                    for j, k in enumerate(demon_index):
                        if k < 0:
                            break # once proceed with <EOS> just jump it away
                        demon_question = demon_data['question'].iloc[k]
                        demon_response = demon_data['answer'].iloc[k]
                        request_text = demon_question
                        messages.insert(-1, {"role": "user", "content":request_text})
                        messages.insert(-1, {"role": "assistant", "content":demon_response})
                        
                    if 'mistralai' in llm_model:
                        messages_mistra = [{"role":"user", "content":'\n'.join([x['content'] for x in messages[:2]])}] + messages[2:]
                        messages_list.append((i, messages_mistra))
                    else:
                        messages_list.append((i, messages))

                batch_response = llm_agent.generate(messages_list)

                print('start to calcualte the reward...')

                batch_response = [y[1] for y in sorted(batch_response, key=lambda x: x[0])]

                model_value = []
                for x in batch_response:
                    try:
                        model_value.append(eval(re.findall(r'####? ?\$?([\+\-]?\d+\.?\d*)', x)[0]))
                    except:
                        model_value.append(None)
                human_value = [eval(re.findall(r'#### ?([\+\-]?\d+)', x)[0]) for x in batch_data['answer']]
                reward_mtrx = torch.Tensor([x==y for x,y in zip(model_value, human_value)]).float().to(device)
                reward_mtrx = reward_mtrx[:,None].repeat(1,fewshot_maxsize)

                batch_length = (batch_index >= 0).sum(dim=1)
                batch_correct = reward_mtrx[:,-1]
                
                print('Reticl Accuracy: {:.4f} - Batch Demon Length: {:.2f}'.format(
                    batch_correct.mean().item(), batch_length.float().mean().item())
                )

                # part 0 intermediate step reward 
                rewards_0 = reward_mtrx * 2 - 1
                rewards_0[:,:-1] = 0
                
                ppo_agent.memory.rewards = rewards_0

                batch_reward = (ppo_agent.memory.rewards * ppo_agent.memory.action_valid).sum() / ppo_agent.memory.action_valid.sum()

                print('start to update the policy...')
                ppo_agent.update()
                batch_progress.set_description("Epoch {:0>2d} - Reward: {:.4f}".format(epoch, batch_reward.item()))

            if epoch % back_every_steps == 0:

                print('save the model...')
                
                save_path = os.path.join(save_root, 'epoch_{}.pt'.format(epoch))
                ppo_agent.save(save_path)

        save_path = os.path.join(save_root, 'last_epoch.pt'.format(epoch))
        ppo_agent.save(save_path)


    if args.do_eval:
        for save_name in os.listdir(save_root):
            if save_name[-3:] != '.pt':
                continue

            save_path = os.path.join(save_root, save_name)
            ppo_agent.load(save_path)
            ppo_agent.policy_old.eval()

            model_values, human_values, agent_action = [], [], []
            
            for batch_data in tqdm(valid_loader, desc="Inferring Batch"):
                qd_mask = torch.ones(len(batch_data['question']), len(demon_data)).to(device)
                qq_input = batch_data['question_embed'].to(device)
                dq_input = demon_question_embed

                
                with torch.no_grad():

                    print('start to generate action...')
                    actions, _, _, _ = ppo_agent.policy_old.generate(qq_input, dq_input, qd_mask, fewshot_maxsize, do_sample=False)
                    batch_index = actions.argmax(dim=2) - 1
                    batch_length = (batch_index >= 0).sum(dim=1)

                    print('Batch Demon Length: {:.2f}'.format(batch_length.float().mean().item()))
                    agent_action.append(batch_index)

                print('start to interact with LLM...')
                messages_list = []
                for i in range(batch_index.shape[0]):
                    query_question = batch_data['question'][i]
                    request_text = query_question
                    messages = prepare_zerohot(request_text, 'cot')

                    demon_index = batch_index[i].clone().detach().cpu().numpy()
                    for j, k in enumerate(demon_index):
                        if k < 0:
                            break # once proceed with <EOS> just jump it away
                        demon_question = demon_data['question'].iloc[k]
                        demon_response = demon_data['answer'].iloc[k]
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

            agent_action = torch.cat(agent_action).detach().cpu().numpy()
            check_path = os.path.join(save_root, save_name+'-agent_action.npy')
            np.save(check_path, agent_action)
            print('{} agent select actions save to {}'.format(save_name, check_path))

            print('='*20,os.path.basename(llm_model),'='*20)
            acc = (model_values == human_values).mean()
            print('Accuracy: {:.4f}'.format(acc))