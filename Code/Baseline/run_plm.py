import re
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.utils import textlist_openaiembed, textlist_sbertembed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        
if __name__ == "__main__":

    num_epochs = 20
    num_workers = 4
    batch_size = 4
    learning_rate = 0.0001

    # model_name = "google-bert/bert-large-uncased"
    model_name = "t5-large"
    # model_name = "roberta-large"

    device = 'cuda'
    
    label_path = '../Data_Paper/test_sample_data_0509_split_0-reinforce-0514.tsv'
    label_data = pd.read_csv(label_path, sep='\t')

    train_data = label_data[label_data['demon_flag']==1]
    valid_data = label_data[label_data['demon_flag']==-1]
    test_data  = label_data[label_data['demon_flag']==0]

    train_data = train_data[['query_instruction_english','problem_clean_english','expert_judge']].to_dict('records')
    valid_data = valid_data[['query_instruction_english','problem_clean_english','expert_judge']].to_dict('records')
    test_data = test_data[['query_instruction_english','problem_clean_english','expert_judge']].to_dict('records')
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto").to(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_metric, save_metric = None, None # best is for valid and save is for test
    save_result = None
    for epoch in range(num_epochs):
        model.train()
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for batch_data in progress:
            inputs = tokenizer(list(zip(batch_data['query_instruction_english'],
                                        batch_data['problem_clean_english'])), return_tensors="pt", padding=True).to(device)
            labels = batch_data['expert_judge'].to(device)
            loss = model(**inputs, labels=labels).loss

            model.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, loss.cpu().detach().numpy()))
        
        model.eval()
        with torch.no_grad():
            pred_labels, true_labels = [], []
            for batch_data in valid_loader:
                inputs = tokenizer(list(zip(batch_data['query_instruction_english'],
                                            batch_data['problem_clean_english'])), return_tensors="pt", padding=True).to(device)
                pred_logits = model(**inputs).logits.cpu().numpy()
                pred_labels.append(np.argmax(pred_logits, axis=1))
                true_labels.append(batch_data['expert_judge'].numpy())
            
            pred_labels = np.concatenate(pred_labels)
            true_labels = np.concatenate(true_labels)

            acc = metrics.accuracy_score(true_labels, pred_labels)
            pre = metrics.precision_score(true_labels, pred_labels)
            rec = metrics.recall_score(true_labels, pred_labels)
            f1 = 2 * pre * rec / (pre + rec)
            if (pre + rec) == 0:
                f1 = -1

            print('Valid - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(acc, pre, rec, f1))

            if best_metric is None or best_metric < f1:
                best_metric = f1
                print('Find new best model...')
                
                pred_labels, true_labels = [], []
                for batch_data in test_loader:
                    inputs = tokenizer(list(zip(batch_data['query_instruction_english'],
                                                batch_data['problem_clean_english'])), return_tensors="pt", padding=True).to(device)
                    pred_logits = model(**inputs).logits.cpu().numpy()
                    pred_labels.append(np.argmax(pred_logits, axis=1))
                    true_labels.append(batch_data['expert_judge'].numpy())
                
                pred_labels = np.concatenate(pred_labels)
                true_labels = np.concatenate(true_labels)

                acc = metrics.accuracy_score(true_labels, pred_labels)
                pre = metrics.precision_score(true_labels, pred_labels)
                rec = metrics.recall_score(true_labels, pred_labels)
                f1 = 2 * pre * rec / (pre + rec)
                if (pre + rec) == 0:
                    f1 = -1
                print('Sample Count: Total: {} - Positive: {} - Negative: {}'.format(len(true_labels), true_labels.sum(), (1-true_labels).sum()))
                print('Test - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(acc, pre, rec, f1))
                save_metric = (acc, pre, rec, f1)
                save_result = pred_labels

    print('Best Saved Model')
    print('Test - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(*save_metric))
