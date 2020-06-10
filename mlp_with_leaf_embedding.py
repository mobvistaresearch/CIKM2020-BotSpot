
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
import os
import torch.nn.utils as utils
from sys import argv
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle

Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
date_name = argv[1]


def recall_at_90_precision(hidden_vec, labels):
    global NUM_VALID
    try:
        hidden_vec = hidden_vec.view(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    except:
        pass
    rec = []
    for j in np.arange(0.01, 0.999, 0.0015):
        # print (j)
        y_pred = hidden_vec > j
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        # print (p,r)
        rec.append((j, p, r))
    rec_90 = sorted(rec, key=lambda x: abs(x[1] - 0.9))
    rec_85 = sorted(rec, key=lambda x: abs(x[1] - 0.85))
    rec_80 = sorted(rec, key=lambda x: abs(x[1] - 0.8))
    print('90% precision:', rec_90[0])
    print('85% precision:', rec_85[0])
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
    X_train, X_test, y_train, y_test = train_test_split(edge_dataset_np, labels, test_size=0.3)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    if train_gbm:
        print('training GBM')
        chan = X_train[:, 0]
        dev = X_train[:, 1]
        labels = X_train[:, 2]
        chan_data = X_more_feats[chan][:, :21]
        dev_data = X_more_feats[dev]
        edge_data = torch.cat((chan_data, dev_data), dim=1)
        edge_data = edge_data.numpy()
        labels = labels.numpy()

        params = {'boosting_type': ['gbdt'], 'max_depth': [4], 'n_estimators': [200],
                  'feature_fraction': [0.8]}

        lgbm = lgb.LGBMClassifier(objective='binary')
        clf = GridSearchCV(lgbm, params, cv=3, scoring='roc_auc')
        clf.fit(edge_data, labels)

        f = open(f'{date_name}/gbm_models/gbm_200_trees_model', "wb")
        pickle.dump(clf.best_estimator_, f)

    return X, X_more_feats, edge_dataset, X_train, X_test, split_num

class NN_with_leaf_emb(torch.nn.Module):

    def __init__(self, X, X_more_feats, *emb_matrixs, gbm_model=None):
        super(NN_with_leaf_emb, self).__init__()
        self.carrier_emb, self.language_emb, self.device_brand_emb, self.plat_os_emb, self.channel_id_emb = emb_matrixs
        self.X = X
        self.X_more_feats = X_more_feats
        self.gbm_best_model = gbm_model
        leaf_dim = 64
        self.nn_model = nn.Sequential(
            nn.Linear(42 + 64, 36),
            nn.ReLU(),
            nn.Linear(36, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

        self.leaf_emb_models = nn.ModuleList()
        for n in range(gbm_model.n_estimators):
            self.leaf_emb_models.append(nn.Embedding(31, leaf_dim))

    def forward(self, edge_data):
        chan = edge_data[:, 0]
        dev = edge_data[:, 1]
        labels = edge_data[:, 2]
        chan_data = self.X[chan][:, :20].to(Device)
        dev_data = self.X[dev].to(Device)
        leaf_feats = self.get_leaf_from_light_gbm(chan, dev)
        final_feats = torch.cat((chan_data, dev_data), dim=1)
        final_feats = torch.cat((final_feats, leaf_feats), dim=1)
        h = self.nn_model(final_feats)
        return h, labels

    def get_leaf_from_light_gbm(self, left_vertices, right_vertices):
        # get leaf indices from gbm model and embed into dense matrix
        output_leaf_emb = []
        chan_data = self.X_more_feats[left_vertices][:, :21]
        dev_data = self.X_more_feats[right_vertices]
        edge_data = torch.cat((chan_data, dev_data), dim=1)
        edge_data = edge_data.numpy()
        # N = len(left_vertices)
        pred_leaf = self.gbm_best_model.predict_proba(edge_data, pred_leaf=True)
        pred_leaf = torch.from_numpy(pred_leaf).long().to(Device)
        if self.gbm_best_model.n_estimators == 1:
            pred_leaf = pred_leaf.view(-1, 1)
            return self.leaf_emb_models[0](pred_leaf[:, 0]).to(Device)

        for i in range(pred_leaf.shape[1]):
            # print (self.leaf_emb_models[i](pred_leaf[:, i]).shape)
            output_leaf_emb.append(self.leaf_emb_models[i](pred_leaf[:, i]).unsqueeze(1))
        ret = torch.cat(output_leaf_emb, dim=1).to(Device)
        ret = torch.mean(ret, dim=1) # mean pooling as the same in botspot

        return ret


X, X_more_feats, edge_dataset, X_train, X_test, split_num = make_dataset(True)
"""
X:feature matrix for channel-campaign node and device node
X_more_feats:feature matrix for channel-campaign node and device node, the features are un-normalized
edge_dataset: triplet:<channel-campaign node, device node, label> for each edge and the label for the edge
X_train: training set split from edge_dataset
X_test: test set split from edge_dataset
split_num: split num to indicate device features or channel-campaign features in X and X_more_feats
"""


chan = X_train[:, 0]
dev = X_train[:, 1]
labels = X_train[:, 2]
X = torch.from_numpy(np.load(f'{date_name}/prepared_data/X_with_feats_selection.npy'))

channel_id_max = int(X_more_feats[split_num - 1:, 20].max().item())
install_carrier_max = int(X_more_feats[:split_num - 1, 5].max().item())
install_language_max = int(X_more_feats[:split_num - 1, 6].max().item())
device_brand_max = int(X_more_feats[:split_num - 1, 7].max().item())
plat_os_max = int(X_more_feats[:split_num - 1, 9].max().item())
print(channel_id_max, install_carrier_max, install_language_max, device_brand_max, plat_os_max)


for g in os.listdir(f"{date_name}/gbm_models/"):
    if '200' in g:
        f = open(f"{date_name}/gbm_models/"+g,"rb")
        gbm_best_model = pickle.load(f)
    break
print (gbm_best_model)

channel_id_emb = nn.Embedding(channel_id_max + 1, 32)
carrier_emb = nn.Embedding(install_carrier_max + 1, 32)
language_emb = nn.Embedding(install_language_max + 1, 32)
device_brand_emb = nn.Embedding(device_brand_max + 1, 32)
device_name_emb = nn.Linear(1 + 1, 12)
plat_os_emb = nn.Embedding(plat_os_max + 1, 8)
nn_model = NN_with_leaf_emb(X,X_more_feats,carrier_emb,language_emb,device_brand_emb,plat_os_emb,channel_id_emb,gbm_model=gbm_best_model)
Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(Device)

train_dset = TensorDataset(X_train)
test_dset = TensorDataset(X_test)
train_data_loader = DataLoader(train_dset, batch_size=256,shuffle = True)
test_data_loader = DataLoader(test_dset, batch_size=256)
optimizer = optim.Adam(nn_model.parameters(), lr=5e-5, weight_decay=2e-6)
epoch = 7


for e in range(epoch):
    print (f"Current Epoch: {e}")
    total_loss = 0.
    val_preds = []
    val_labels = []
    for index,d in enumerate(train_data_loader):
        nn_model.to(Device)
        nn_model.train()
        h,labels = nn_model(d[0])
        labels = labels.to(Device)
        loss = torch.sum(-1 * torch.log(h[labels == 1]))
        N = torch.sum(labels == 1).item()
        neg_loss = torch.sort(-1. * torch.log(1 - h[labels == 0]).view(-1, ), descending=True)[0][:3 * N]
        neg_N = len(neg_loss)
        loss += torch.sum(neg_loss)
        loss = loss / (N + neg_N)
        loss.backward()
        utils.clip_grad_value_(nn_model.parameters(), 4)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        current_loss = total_loss / (index + 1)
        if index % 50 == 0:
            print(f'current loss for index:{index} is: {current_loss}')
    if e%2==0:
        for d in test_data_loader:
            with torch.no_grad():
                nn_model.to(Device)
                nn_model.eval()
                h,labels = nn_model(d[0])
                val_preds.extend(list(h.cpu().numpy()))
                val_labels.extend(list(labels.cpu().numpy()))
        print (f'the Recall@T Precision is: ')
        pr_score = recall_at_90_precision(np.asarray(val_preds), np.asarray(val_labels))
        # with open(f'{date_name}/baseline_records/nn_with_leaf_emb','wb') as f:
        #     pickle.dump(pr_score,f)

