import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.generator import LocalGenerator, RemoteGenerator
from Tools.utils import textlist_openaiembed, prepare_zerohot

PROCESS_NUM=40
EMBED_MODEL_NAME='text-embedding-3-small'
EMBED_BATCH_SIZE=256


def update_fewshot(zeroshot_messages, fewshot_select):
    for j in range(len(fewshot_select)):
        request_text = fewshot_select.iloc[j]['question']
        zeroshot_messages.insert(-1, {"role": "user", "content":request_text})
        answer_text = fewshot_select.iloc[j]['answer']
        zeroshot_messages.insert(-1, {"role": "assistant", "content":answer_text})
    return zeroshot_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--prompt_style", type=str, default='cot',
                        choices=['cot',])
    parser.add_argument("--fewshot_flag", type=str, default='zero', 
                        choices=['qrandom','qsim',])
    parser.add_argument("--fewshot_size", type=int, default=0)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_cache",  action='store_true', help='search and insert cache in local machine.')
    parser.add_argument("--search_cache", action='store_true', help='search cache for LLM inference result.')
    parser.add_argument("--insert_cache", action='store_true', help='insert the LLM result into cache.')
    parser.add_argument("--run_id", type=str, default=None, help='runid, used for multiple run and average result')
    args = parser.parse_args()

    model_name = args.model_name
    prompt_style = args.prompt_style
    batch_size = args.batch_size
    
    fewshot_flag = args.fewshot_flag
    fewshot_size = args.fewshot_size

    local_cache = args.local_cache
    search_cache = args.search_cache
    insert_cache = args.insert_cache

    label_path = args.label_path
    label_data = pd.read_csv(label_path, sep='\t')

    test_data = label_data[label_data['demon_flag']==0].reset_index()
    fewshot_data = label_data[label_data['demon_flag']==1].reset_index()
    
    if 'gpt' in model_name:
        llm_agent = RemoteGenerator(model_name, PROCESS_NUM, search_cache, insert_cache, local_cache)
    else:
        llm_agent = LocalGenerator(model_name, batch_size, PROCESS_NUM, 
                                   search_cache=search_cache, insert_cache=insert_cache, local_cache=local_cache)

    if fewshot_flag != 'zero':
        
        if fewshot_flag == 'qsim':
            demon_list = fewshot_data['question'].tolist()
            demon_embed = textlist_openaiembed(demon_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            demon_norm = np.linalg.norm(demon_embed, 2, axis=1)

            query_list = test_data['question'].tolist()
            query_embed = textlist_openaiembed(query_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            query_norm = np.linalg.norm(query_embed, 2, axis=1)

    messages_list = []
    for i in range(len(test_data)):

        request_text = test_data.iloc[i]['question']

        system_text = "Your are a helpful assistant. Please provide correct solution to the given math word problem. The solution value should be provided at the end of the reponse, and in the format like #### X, where X is solution value."
        
        messages = [{'role':'system', 'content':system_text},
                    {'role':'user', 'content':request_text},]
        
        if fewshot_flag == 'qrandom':
            # prepare for the fewshot learning (same query question simularity)
            fewshot_select = fewshot_data.sample(fewshot_size)
        
        elif fewshot_flag == 'qsim':
            embed_scores = np.matmul(query_embed[i], demon_embed.T) / (query_norm[i] * demon_norm)
            # pick out the top ones
            fewshot_select = np.argsort(embed_scores)[::-1][:fewshot_size]
            fewshot_select = fewshot_data.iloc[fewshot_select]

        if fewshot_flag != 'zero':
            # add fewshot samples to message
            messages = update_fewshot(messages, fewshot_select)
        
        if 'mistralai' in model_name or 'claude' in model_name:
            messages = [{"role":"user", "content":'\n'.join([x['content'] for x in messages[:2]])}] + messages[2:]
        
        messages_list.append((i, messages))

    response_list = llm_agent.generate(messages_list)
    response_list = [y[1] for y in sorted(response_list, key=lambda x: x[0])]
    model_respond = response_list
    model_value = []
    for x in model_respond:
        try:
            model_value.append(eval(re.findall(r'####? ?\$?([\+\-]?\d+\.?\d*)', x)[0]))
        except:
            model_value.append(None)
    human_value = [eval(re.findall(r'#### ?([\+\-]?\d+)', x)[0]) for x in test_data['answer']]

    # prepare output data information
    test_data['human_value'] = human_value
    test_data['model_value'] = model_value
    test_data['model_respond'] = model_respond
    save_path = '../../Save/{}-{}-{}-{}shot-rid{}.tsv'.format(
        os.path.basename(label_path).split('.tsv')[0],
        os.path.basename(model_name).lower(), prompt_style,
        fewshot_flag + ('' if fewshot_flag=='zero' else '_{}'.format(str(fewshot_size))), 
        args.run_id
    )
    print('result save to ', save_path)
    test_data.to_csv(save_path, index=False, sep='\t')
    
    if args.do_eval:
        
        print('='*20,os.path.basename(model_name),'='*20)
        acc = (np.array(model_value) == np.array(human_value)).mean()
        print('Accuracy: {:.4f}'.format(acc))

    