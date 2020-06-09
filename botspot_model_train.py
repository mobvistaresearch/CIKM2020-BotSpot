from sklearn.feature_selection import *
from sklearn.ensemble import *
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
import math
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer,auc,roc_auc_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import  confusion_matrix,accuracy_score
import os
import torch.nn.utils as utils


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
def recall_at_90_precision(hidden_vec, labels):
    # a utility function to obtain recall and precision for various probability threshold
    global NUM_VALID
    try:
        hidden_vec = hidden_vec.view(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    except:
        pass
    rec = []
    for j in np.arange(0.005, 0.999, 0.001):
        # print (j)
        y_pred = hidden_vec > j
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        # print (p,r)
        rec.append((j,p, r))
    rec_90 = sorted(rec, key=lambda x: abs(x[1] - 0.9))
    rec_85 = sorted(rec, key=lambda x: abs(x[1] - 0.85))
    rec_80 = sorted(rec, key=lambda x: abs(x[1] - 0.8))
    print ('90% precision:', rec_90[0])
    print ('85% precision:', rec_85[0])
    print('80% precision:', rec_80[0])
    return rec

def eval_model(model,data_loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for edges in data_loader:
            if np.random.uniform()>0.85:
                print (f'current progress: {len(preds)}, out of total of {len(data_loader.dataset)}')
            hidden_vec, label = model(edges[0])
            preds.extend(list(hidden_vec.view(-1,).cpu().numpy()))
            labels.extend(list(label.view(-1,).cpu().numpy()))
    preds, labels = np.asarray(preds),np.asarray(labels)
    rec = recall_at_90_precision(preds, labels)
    return rec



def train(model, data_loader, data_loader_test,optimizer, epoch=50, thres=0.5, weight=1, agg='mean', save_name = None,multilayer = True,local = False,graphsage = False):
    for i in range(epoch):
        min_loss = 10.
        total_loss = 0
        print ('epoch: ', i)
        for index, d in enumerate(data_loader):
            try:
                model.train()
                model.to(Device)
                hidden_vec, label = model(d[0])
                loss = torch.sum(-1. * torch.log(hidden_vec[label == 1]))
                N = torch.sum(label == 1).item()
                # hard negative mining for loss back prop
                neg_loss = torch.sort(-1. * torch.log(1 - hidden_vec[label == 0]).view(-1, ), descending=True)[0][:3*N]
                neg_N = len(neg_loss)
                loss += torch.sum(neg_loss)
                loss = loss / (N+neg_N)

                # hinge loss
                # loss_pos = 1 - hidden_vec[label == 1]
                # N_pos = torch.sum(loss_pos>0)
                # loss_pos = torch.sum(loss_pos[loss_pos > 0])

                # loss_neg = 1 - (-1. * hidden_vec[label == 0])
                # loss_neg = torch.sort(loss_neg[loss_neg > 0].view(-1, ), descending=True)[0]
                # try:
                #     loss_neg = torch.sum(loss_neg[:2*N_pos])
                #     N_neg = 2*N_pos
                # except:
                #     N_neg = len(loss_neg)
                #     loss_neg = torch.sum(loss_neg)

                # loss = (loss_neg + loss_pos) /(N_pos+N_neg)
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    print (loss.item())
                    continue

                loss.backward()
                utils.clip_grad_value_(model.parameters(), 4)
                optimizer.step()
                optimizer.zero_grad()

                # if index % 150 == 0 and index > 0:
                #     path = 'saved_models/newest_model/model.pt'
                #
                #     torch.save(model.state_dict(), path)
                #     path = 'saved_models/newest_model/optim.pt'
                #     torch.save(optimizer.state_dict(), path)
                total_loss += loss.item()
                current_loss = total_loss/(index + 1)

                if index % 50 == 0:
                    print (f'current loss for index:{index} is: {current_loss}')

                # num_eval += 1
                # print ('num_eval:', num_eval)
                # print (f'training loss for epoch:{i} is: {total_loss/(index +1)}')
            except:
                continue
        path = f'{save_name}/saved_models/botspot_epoch_{i}.pt' if multilayer else f'{save_name}/saved_models/botspot_single_layer_epoch_{i}.pt'
        if local:
            path = f'{save_name}/saved_models/botspot_local_checkpoints.pt'
        if graphsage:
            path = f'{save_name}/saved_models/botspot_graphsage_checkpoints.pt'
        torch.save(model.state_dict(),path)
        pr_rec = eval_model(model,data_loader_test)
        path = f'{save_name}/pr_records/pr_rec_multilayer.pickle' if multilayer else f'{save_name}/pr_records/pr_rec_botspot_checkpoints.pickle'
        if local:
            path = f'{save_name}/pr_records/pr_rec_local_checkpoints.pickle'
        if graphsage:
            path = f'{save_name}/pr_records/pr_rec_graphsage_checkpoints.pickle'
        with open(path,'wb') as f:
            pickle.dump(pr_rec,f)


