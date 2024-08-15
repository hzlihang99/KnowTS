import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import date
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.generator import LocalGenerator, RemoteGenerator
from Tools.utils import textlist_openaiembed, text2prompt, prepare_zerohot

PROCESS_NUM=40
EMBED_MODEL_NAME='text-embedding-3-small'
EMBED_BATCH_SIZE=256


def update_fewshot(zeroshot_messages, fewshot_select):
    for j in range(len(fewshot_select)):
        request_text = text2prompt(
            fewshot_select['query_instruction_english'].iloc[j],
            fewshot_select['problem_clean_english'].iloc[j]
        )
        zeroshot_messages.insert(-1, {"role": "user", "content":request_text})
        answer_text = fewshot_select.iloc[j]['expert_reason']
        zeroshot_messages.insert(-1, {"role": "assistant", "content":answer_text})
    return zeroshot_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--prompt_style", type=str, default='cot',
                        choices=['regular', 'reason', 'cot'])
    parser.add_argument("--fewshot_set", type=str, default='mix',
                        choices=['mix','pos','neg'])
    parser.add_argument("--fewshot_flag", type=str, default='zero', 
                        choices=['ksame_qrandom','ksame_qsim',])
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
    
    fewshot_set = args.fewshot_set
    fewshot_flag = args.fewshot_flag
    fewshot_size = args.fewshot_size

    local_cache = args.local_cache
    search_cache = args.search_cache
    insert_cache = args.insert_cache

    label_path = args.label_path
    label_data = pd.read_csv(label_path, sep='\t')

    test_data = label_data[label_data['demon_flag']==0].reset_index()
    fewshot_data = label_data[label_data['demon_flag']==1].reset_index()
    if fewshot_set == 'pos':
        fewshot_data = fewshot_data[fewshot_data['expert_judge']==1].reset_index()
    elif fewshot_set == 'neg':
        fewshot_data = fewshot_data[fewshot_data['expert_judge']==0].reset_index()

    if 'gpt' in model_name:
        llm_agent = RemoteGenerator(model_name, PROCESS_NUM, search_cache, insert_cache, local_cache)
    else:
        llm_agent = LocalGenerator(model_name, batch_size, PROCESS_NUM, 
                                   search_cache=search_cache, insert_cache=insert_cache, local_cache=local_cache)

    if fewshot_flag != 'zero':
        
        if fewshot_flag == 'both_sim':
            demon_list = fewshot_data.apply(lambda x: text2prompt(
                x['query_instruction_english'],x['problem_clean_english']
            ), axis=1).to_list()
            demon_embed = textlist_openaiembed(demon_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            demon_norm = np.linalg.norm(demon_embed, 2, axis=1)

            query_list = test_data.apply(lambda x: text2prompt(
                x['query_instruction_english'],x['problem_clean_english']
            ), axis=1).to_list()
            query_embed = textlist_openaiembed(query_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            query_norm = np.linalg.norm(query_embed, 2, axis=1)

        elif fewshot_flag == 'ksame_qsim':
            demon_list = fewshot_data['problem_clean_english'].tolist()
            demon_embed = textlist_openaiembed(demon_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            demon_norm = np.linalg.norm(demon_embed, 2, axis=1)

            query_list = test_data['problem_clean_english'].tolist()
            query_embed = textlist_openaiembed(query_list, EMBED_MODEL_NAME, EMBED_BATCH_SIZE)
            query_norm = np.linalg.norm(query_embed, 2, axis=1)

    messages_list = []
    for i in range(len(test_data)):

        request_text = text2prompt(
            test_data['query_instruction_english'].iloc[i],
            test_data['problem_clean_english'].iloc[i]
        )
        
        # prepare the zeroshot message
        messages = prepare_zerohot(request_text, prompt_style)

        if fewshot_flag == 'both_random':
            # prepare for the fewshot learning (total random)
            # total select the random result from fewshot data
            fewshot_select = fewshot_data.sample(fewshot_size)
        
        elif fewshot_flag == 'ksame_qrandom':
            # prepare for the fewshot learning (same query question simularity)
            fewshot_select = fewshot_data[fewshot_data['tag_code']==test_data['tag_code'].iloc[i]]
            fewshot_select = fewshot_select.sample(fewshot_size)
        
        elif fewshot_flag == 'both_sim':
            embed_scores = np.matmul(query_embed[i], demon_embed.T) / (query_norm[i] * demon_norm)
            # pick out the top ones
            fewshot_select = np.argsort(embed_scores)
            fewshot_select = fewshot_select[::-1]
            fewshot_select = fewshot_data.iloc[fewshot_select[:fewshot_size]]
        
        elif fewshot_flag == 'ksame_qsim':
            demon_select = fewshot_data[fewshot_data['tag_code']==test_data['tag_code'].iloc[i]].index.to_numpy()
            embed_scores = np.matmul(query_embed[i], demon_embed[demon_select].T) / (query_norm[i] * demon_norm[demon_select])
            # pick out the top ones
            fewshot_select = np.argsort(embed_scores)[::-1]
            fewshot_select = fewshot_data.iloc[demon_select[fewshot_select[:fewshot_size]]]

        if fewshot_flag != 'zero':
            # add fewshot samples to message
            messages = update_fewshot(messages, fewshot_select)


        if 'mistralai' in model_name or 'claude' in model_name:
            messages = [{"role":"user", "content":'\n'.join([x['content'] for x in messages[:2]])}] + messages[2:]
        
        messages_list.append((i, messages))

    response_list = llm_agent.generate(messages_list)
    response_list = [y[1] for y in sorted(response_list, key=lambda x: x[0])]
    model_judge = np.array([int(bool(re.findall(r'<Yes>',x))) for x in response_list])
    model_reason = response_list
    
    if args.do_eval:
        expert_judge = test_data['expert_judge']
        if 'demon_flag' in test_data:
            test_flag = test_data['demon_flag'].to_numpy() == 0
        else:
            test_flag = np.array([True,]*len(expert_judge))

        print('='*20,os.path.basename(model_name),'='*20)
        print('Sample Count: Total: {} - Positive: {} - Negative: {}'.format(
            len(expert_judge[test_flag]), (expert_judge[test_flag]==1).sum(), 
            (expert_judge[test_flag]==0).sum()
        ))
        acc = metrics.accuracy_score(y_true=list(expert_judge[test_flag]),y_pred=list(model_judge[test_flag]))
        pre = metrics.precision_score(y_true=list(expert_judge[test_flag]),y_pred=list(model_judge[test_flag]))
        rec = metrics.recall_score(y_true=list(expert_judge[test_flag]),y_pred=list(model_judge[test_flag]))
        f1 = 2 * pre * rec / (pre + rec)
        print('Accuracy: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}'.format(acc, pre, rec, f1))

    # prepare output data information
    test_data['model_judge'] = model_judge
    test_data['model_reason'] = model_reason

    save_path = '../../Save/{}-{}-{}-{}-{}shot-rid{}.tsv'.format(
        os.path.basename(label_path).split('.tsv')[0], fewshot_set,
        os.path.basename(model_name).lower(), prompt_style,
        fewshot_flag + ('' if fewshot_flag=='zero' else '_{}'.format(str(fewshot_size))), 
        args.run_id
    )
    print('result save to ', save_path)
    test_data.to_csv(save_path, index=False, sep='\t')