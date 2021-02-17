from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.utils as utils
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
    for j in np.arange(0.01, 0.99, 0.001):
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

class NN(torch.nn.Module):

    def __init__(self, X):
        super(NN,self).__init__()
        self.X = X
        self.nn_model = nn.Sequential(
            nn.Linear(42, 24),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self,edge_data):
        chan = edge_data[:, 0]
        dev = edge_data[:, 1]
        labels = edge_data[:, 2]
        chan_data = self.X[chan][:, :20].to(Device)
        dev_data = self.X[dev].to(Device)
        edge_data = torch.cat((chan_data, dev_data), dim=1)
        h = self.nn_model(edge_data)
        return h,labels


if __name__ == '__main__':
    X, X_more_feats, edge_dataset, X_train, X_test, split_num = make_dataset(train_gbm=False)
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
    X = torch.from_numpy(np.load(f'{date_name}/prepared_data/X_with_feats_selection.npy')) # remove several redundant features
    nn_model = NN(X)
    Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device=Device
    print(Device)

    train_dset = TensorDataset(X_train)
    test_dset = TensorDataset(X_test)
    train_data_loader = DataLoader(train_dset, batch_size=256,shuffle = True)
    test_data_loader = DataLoader(test_dset, batch_size=256)
    optimizer = optim.Adam(nn_model.parameters(), lr=2e-4, weight_decay=1e-6)
    epoch = 10
    for e in range(epoch):
        total_loss = 0.
        val_preds = []
        val_labels = []
        for index,d in enumerate(train_data_loader):
            nn_model.to(device)
            nn_model.train()
            h,labels = nn_model(d[0])
            labels = labels.to(device)
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

            if index % 100 == 0:
                print(f'current loss for index:{index} is: {current_loss}')
        if e%3==0 and e>0:
            for d in test_data_loader:
                with torch.no_grad():
                    nn_model.to(device)
                    nn_model.eval()
                    h,labels = nn_model(d[0])
                    val_preds.extend(list(h.cpu().numpy()))
                    val_labels.extend(list(labels.cpu().numpy()))
            print (f"epoch:{e}:precision recall score is: \n")

            pr_score = recall_at_90_precision(np.asarray(val_preds), np.asarray(val_labels))


