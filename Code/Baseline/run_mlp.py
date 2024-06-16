import os
import sys
import argparse
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.class_module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        logits = self.class_module(x)
        return logits
        

class TaskData(object):
    def __init__(self, kx, qx, y):
        self.kx = kx
        self.qx = qx
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {'kx':torch.Tensor(self.kx[index]).float(),
                'qx':torch.Tensor(self.qx[index]).float(),
                'y':torch.Tensor([self.y[index]]).long()}


class TaskDataWrap(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {'x':torch.FloatTensor(self.x[index]),
                'y':torch.LongTensor([self.y[index]])}

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--embed_name', type=str, required=True,
                        choices=['sbert', 'openai'])
    args = parser.parse_args()

    num_epochs = 100
    num_workers = 2
    batch_size = 128
    learning_rate = 0.001

    device = 'cuda:0'

    label_path = args.label_path
    label_data = pd.read_csv(label_path, sep='\t')

    train_data = label_data[label_data['demon_flag']!=0]
    test_data  = label_data[label_data['demon_flag']==0]

    train_knowledge_list = train_data['query_instruction_english'].tolist()
    if args.embed_name == 'sbert':
        train_knowledge_embed = textlist_sbertembed(train_knowledge_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        train_knowledge_embed = textlist_openaiembed(train_knowledge_list, "text-embedding-3-small")
    
    train_question_list = train_data['problem_clean_english'].tolist()
    if args.embed_name == 'sbert':
        train_question_embed = textlist_sbertembed(train_question_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        train_question_embed = textlist_openaiembed(train_question_list, "text-embedding-3-small")
    train_label = train_data['expert_judge'].tolist()

    test_knowledge_list = test_data['query_instruction_english'].tolist()
    if args.embed_name == 'sbert':
        test_knowledge_embed = textlist_sbertembed(test_knowledge_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        test_knowledge_embed = textlist_openaiembed(test_knowledge_list, "text-embedding-3-small")
    
    test_question_list = test_data['problem_clean_english'].tolist()
    if args.embed_name == 'sbert':
        test_question_embed = textlist_sbertembed(test_question_list, "sentence-transformers/all-mpnet-base-v2")
    else:
        test_question_embed = textlist_openaiembed(test_question_list, "text-embedding-3-small")
    
    test_label = test_data['expert_judge'].tolist()
    
    train_data = TaskData(train_knowledge_embed,train_question_embed,train_label)
    test_data  = TaskData(test_knowledge_embed, test_question_embed, test_label)
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers)

    model = MLP(768*2, 256, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_metric, save_metric = None, None # best is for valid and save is for test
    save_result = None
    for epoch in range(num_epochs):
        model.train()
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for batch_data in progress:
            kx = batch_data['kx'].to(device)
            qx = batch_data['qx'].to(device)
            x = torch.cat([kx,qx], dim=1)

            pred_logits = model(x)
            true_labels = batch_data['y'][:,0].to(device)

            loss = F.cross_entropy(pred_logits, true_labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, loss.cpu().detach().numpy()))
        
        model.eval()
        with torch.no_grad():
            pred_labels, true_labels = [], []
            for batch_data in test_loader:
                kx = batch_data['kx'].to(device)
                qx = batch_data['qx'].to(device)
                x = torch.cat([kx,qx], dim=1)

                pred_logits = model(x).cpu().numpy()
                pred_labels.append(np.argmax(pred_logits, axis=1))
                true_labels.append(batch_data['y'][:,0].numpy())
            
            pred_labels = np.concatenate(pred_labels)
            true_labels = np.concatenate(true_labels)

            acc = metrics.accuracy_score(true_labels, pred_labels)
            pre = metrics.precision_score(true_labels, pred_labels)
            rec = metrics.recall_score(true_labels, pred_labels)
            f1 = 2 * pre * rec / (pre + rec)
            if (pre + rec) == 0:
                f1 = -1

            print('Test - Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(acc, pre, rec, f1))

            if best_metric is None or best_metric < f1:
                best_metric = f1
                print('Find new best model...')
                save_metric = (acc, pre, rec, f1)
                save_result = pred_labels

    print('Best Saved Model')
    print('Acc: {:.4f} - Pre: {:.4f} - Rec: {:.4f} - F1: {:.4f}'.format(*save_metric))