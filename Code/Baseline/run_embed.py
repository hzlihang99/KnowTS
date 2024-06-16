import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import date
from itertools import groupby
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.utils import textlist_openaiembed, textlist_sbertembed

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--embed_name', type=str, required=True,
                        choices=['sbert', 'openai'])
    args = parser.parse_args()

    label_path = args.label_path
    embed_name = args.embed_name

    label_data = pd.read_csv(label_path, sep='\t')
    
    test_data = label_data[label_data['demon_flag']==0].reset_index(drop=True)
    question_list = test_data['problem_clean_english'].unique().tolist()

    demonstration_data = label_data[label_data['demon_flag']==1].reset_index(drop=True)
    demonstration_data = demonstration_data[demonstration_data['expert_judge']==1].reset_index(drop=True)
    demonstration_list = demonstration_data['problem_clean_english'].tolist()
    demonstration_key = demonstration_data['tag_code'].tolist()

    if args.embed_name == 'sbert':
        question_embed = textlist_sbertembed(question_list, "sentence-transformers/all-mpnet-base-v2")
        demonstration_embed = textlist_sbertembed(demonstration_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        question_embed = textlist_openaiembed(question_list, "text-embedding-3-small")
        demonstration_embed = textlist_openaiembed(demonstration_list, "text-embedding-3-small")
    
    question_norm = np.linalg.norm(question_embed, 2, axis=1)
    demonstration_norm = np.linalg.norm(demonstration_embed, 2, axis=1)

    print('='*20,'query (center) question match K','='*20)
    # Here we use the embedding sum of question as the center location for that knowledge tag
    knowledge_index = sorted([(x, i) for i, x in enumerate(demonstration_key)], key=lambda x: x[0])
    knowledge_group = [(k, np.array([y[1] for y in g])) for k,g in groupby(knowledge_index, key=lambda x: x[0])]

    best_metric, save_result = None, None
    for k in np.arange(1,201,20):
        pred_data = []
        for tag_code, index_list in knowledge_group:
            knowledge_embed = demonstration_embed[index_list].mean(axis=0)
            knowledge_norm = np.linalg.norm(knowledge_embed, 2)
            score_list = np.matmul(knowledge_embed / knowledge_norm, (question_embed / question_norm[:,None]).T)
            threshold = sorted(score_list, reverse=True)[k-1]
            positive_indx = np.where(score_list >= threshold)[0]
            pred_data.extend([{'tag_code':tag_code, 'problem_clean_english':question_list[x], 'model_judge':1} for x in positive_indx])

        pred_data = pd.DataFrame(pred_data)
        pred_data = test_data.merge(pred_data, how='left', on=['tag_code','problem_clean_english'])
        pred_data['model_judge'] = pred_data['model_judge'].fillna(0)
        acc = metrics.accuracy_score(pred_data['expert_judge'], pred_data['model_judge'])
        pre = metrics.precision_score(pred_data['expert_judge'], pred_data['model_judge'])
        rec = metrics.recall_score(pred_data['expert_judge'], pred_data['model_judge'])
        f1 = 2 * pre * rec / (pre + rec)
        print('Top K: {} - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(k-1, acc, pre, rec, f1))
        
        if best_metric == None or f1 > best_metric :
            best_metric = f1
            save_result = pred_data
    
    # save the best resul into a file
    save_path = '../Data_Paper/{}-{}_qq_topk-{}.tsv'.format(
        os.path.basename(label_path).split('.tsv')[0],
        embed_name, date.today().strftime("%m%d")
    )
    save_result.to_csv(save_path, index=False, sep='\t')

    
    print('='*20,'query (center) question match threshold','='*20)

    best_metric, save_result = None, None
    for threshold in np.arange(0,0.85,0.05):
        pred_data = []
        for tag_code, index_list in knowledge_group:
            knowledge_embed = demonstration_embed[index_list].mean(axis=0)
            knowledge_norm = np.linalg.norm(knowledge_embed, 2)
            score_list = np.matmul(knowledge_embed / knowledge_norm, (question_embed / question_norm[:,None]).T)
            positive_indx = np.where(score_list >= threshold)[0]
            pred_data.extend([{'tag_code':tag_code, 'problem_clean_english':question_list[x], 'model_judge':1} for x in positive_indx])

        pred_data = pd.DataFrame(pred_data)
        pred_data = test_data.merge(pred_data, how='left', on=['tag_code','problem_clean_english'])
        pred_data['model_judge'] = pred_data['model_judge'].fillna(0)
        acc = metrics.accuracy_score(pred_data['expert_judge'], pred_data['model_judge'])
        pre = metrics.precision_score(pred_data['expert_judge'], pred_data['model_judge'])
        rec = metrics.recall_score(pred_data['expert_judge'], pred_data['model_judge'])
        f1 = 2 * pre * rec / (pre + rec)
        print('threshold: {:.2f} - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(threshold, acc, pre, rec, f1))

        if best_metric == None or f1 > best_metric :
            best_metric = f1
            save_result = pred_data
    
    save_path = '../Data_Paper/{}-{}_qq_sim-{}.tsv'.format(
        os.path.basename(label_path).split('.tsv')[0],
        embed_name, date.today().strftime("%m%d")
    )
    save_result.to_csv(save_path, index=False, sep='\t')

    print('='*20,'query (text) question match','='*20)

    # knowledge information
    knowledge_data = label_data[['tag_code','query_instruction_english']].drop_duplicates().reset_index(drop=True)
    knowledge_list = knowledge_data['query_instruction_english'].tolist()

    if args.embed_name == 'sbert':
        knowledge_embed = textlist_sbertembed(knowledge_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        knowledge_embed = textlist_openaiembed(knowledge_list, "text-embedding-3-small")
    knowledge_norm = np.linalg.norm(knowledge_embed, 2, axis=1)
    score_mtrx = np.matmul(knowledge_embed / knowledge_norm[:,None], (question_embed / question_norm[:,None]).T)

    best_metric, save_result = None, None
    for threshold in np.arange(0,0.61,0.05):
        pred_data = []
        for i in range(score_mtrx.shape[0]):
            for j in range(score_mtrx.shape[1]):
                if score_mtrx[i][j] >= threshold:
                    pred_data.append({'tag_code':knowledge_data['tag_code'].iloc[i],'problem_clean_english':question_list[j], 'model_judge':1})
        pred_data = pd.DataFrame(pred_data)
        pred_data = test_data.merge(pred_data, how='left', on=['tag_code','problem_clean_english'])
        pred_data['model_judge'] = pred_data['model_judge'].fillna(0)
        acc = metrics.accuracy_score(pred_data['expert_judge'], pred_data['model_judge'])
        pre = metrics.precision_score(pred_data['expert_judge'], pred_data['model_judge'])
        rec = metrics.recall_score(pred_data['expert_judge'], pred_data['model_judge'])
        f1 = 2 * pre * rec / (pre + rec)
        print('threshold: {:.2f} - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(threshold, acc, pre, rec, f1))

        if best_metric == None or f1 > best_metric :
            best_metric = f1
            save_result = pred_data

    save_path = '../../Save/{}-{}_kq_sim-{}.tsv'.format(
        os.path.basename(label_path).split('.tsv')[0],
        embed_name, date.today().strftime("%m%d")
    )
    save_result.to_csv(save_path, index=False, sep='\t')