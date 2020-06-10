from sklearn.feature_selection import *
from sklearn.ensemble import *
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sys import argv
date_name = argv[1]

def recall_at_90_precision(hidden_vec, labels):
    global NUM_VALID
    try:
        hidden_vec = hidden_vec.view(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    except:
        pass
    rec = []
    for j in np.arange(0.005, 0.999, 0.0015):
        # print (j)
        y_pred = hidden_vec > j
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        # print (p,r)
        rec.append((j,p, r))
        if p >= 0.91:
            break
    rec_90 = sorted(rec, key=lambda x: abs(x[1] - 0.9))
    rec_85 = sorted(rec, key=lambda x: abs(x[1] - 0.85))
    rec_80 = sorted(rec, key=lambda x: abs(x[1] - 0.8))
    print ('90% precision:', rec_90[0])
    print ('85% precision:', rec_85[0])
    print('80% precision:', rec_80[0])
    return rec


def make_dataset(train_gbm=False):
    dev_feats_with_device_level_feats = np.load(f"{date_name}/prepared_data/dev_feats.npy")
    dev_feats = np.load(f"{date_name}/prepared_data/dev_normed_feats.npy")
    channel_feats = np.load(f"{date_name}/prepared_data/chan_feats.npy")
    channel_normed_feats = np.load(f"{date_name}/prepared_data/chan_normed_feats.npy")
    edge_index_bots = np.load(f"{date_name}/prepared_data/edge_bots.npy")
    edge_index_normal = np.load(f"{date_name}/prepared_data/edge_normal.npy")

    dev_feat_dim = dev_feats.shape[1]
    channel_normed_dim = channel_normed_feats.shape[1]

    split_num = dev_feats.shape[0] + 1

    padding_channel_1 = np.zeros((channel_normed_feats.shape[0], dev_feat_dim - channel_normed_dim))
    channel_normed_feats = np.hstack((channel_normed_feats, padding_channel_1))

    X = np.vstack((dev_feats, channel_normed_feats))
    X = torch.from_numpy(X).float()

    channel_feats = np.hstack((channel_feats, np.arange(channel_feats.shape[0]).reshape((-1, 1))))

    channel_dim = channel_feats.shape[1]

    dev_feats_with_device_level_feats_dim = dev_feats_with_device_level_feats.shape[1]

    padding_channel_2 = np.zeros((channel_feats.shape[0], dev_feats_with_device_level_feats_dim - channel_dim))
    channel_feats = np.hstack((channel_feats, padding_channel_2))

    X_more_feats = np.vstack((dev_feats_with_device_level_feats, channel_feats))
    X_more_feats = torch.from_numpy(X_more_feats).float()

    edge_dataset = torch.cat((torch.from_numpy(edge_index_bots), torch.from_numpy(edge_index_normal)))
    edge_dataset_np = edge_dataset.numpy()

    labels = edge_dataset_np[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(edge_dataset_np, labels, test_size=0.3, random_state=42)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
        # f = open('light_gbm_models_less_feats/gbm_models_'+str(score)[:5],"wb")
        # pickle.dump(clf.best_estimator_,f, protocol=2)
    return X, X_more_feats, edge_dataset, X_train, X_test, split_num



if __name__ == '__main__':
    X, X_unnormalized, edge_dataset, X_train, X_test, split_num = make_dataset(train_gbm=False)
    chan = X_train[:, 0]
    dev = X_train[:, 1]
    labels = X_train[:, 2]
    chan_data = X[chan][:, :21]
    dev_data =  X[dev]
    edge_data = torch.cat((chan_data, dev_data), dim=1).numpy()

    params = {'boosting_type': ['gbdt'], 'max_depth': [4], 'n_estimators': [300],
              'feature_fraction': [0.8]}

    lgbm = lgb.LGBMClassifier(objective='binary')
    clf = GridSearchCV(lgbm, params, cv=3, scoring='roc_auc')
    clf.fit(edge_data, labels.numpy())
    model = clf.best_estimator_
    chan = X_test[:,0]
    dev =  X_test[:,1]
    labels = X_test[:,2]
    chan_data = X[chan][:,:21]
    dev_data = X[dev]
    edge_data = torch.cat((chan_data,dev_data),dim=1).numpy()
    pred_proba = model.predict_proba(edge_data)[:,1]
    pr_score = recall_at_90_precision(pred_proba, labels.numpy())
    # with open(f'{date_name}/baselines/lightgbm_300_tree.pr_rec','wb') as f:
    #     pickle.dump(pr_score,f)


