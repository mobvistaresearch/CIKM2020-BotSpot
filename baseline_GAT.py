from sklearn.model_selection import train_test_split
from model_main_GAT_baseline import *
from model_train import *
from model_utils import *
from sys import argv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data import TensorDataset,DataLoader
import torch.nn.utils as utils
from torch.utils import data
# from  apex import amp

Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
date_name = argv[1]

def make_dataset(train_gbm=False):
    dev_feats_with_device_level_feats = np.load(f"{date_name}/prepared_data/dev_feats.npy")
    dev_feats = np.load(f"{date_name}/prepared_data/dev_normed_feats.npy")
    channel_feats = np.load(f"{date_name}/prepared_data/chan_feats.npy")
    channel_normed_feats = np.load(f"{date_name}/prepared_data/chan_normed_feats.npy")
    edge_bots = np.load(f"{date_name}/prepared_data/edge_bots.npy")
    edge_normal = np.load(f"{date_name}/prepared_data/edge_normal.npy")
    graph_data = np.vstack((edge_normal,edge_bots))
    print (f"graph data len: {len(graph_data)}")
    df_graph_gr = pd.DataFrame(graph_data[:,:2],columns=['a','b']).groupby(['a'],as_index = False).agg({'b':'count'})
    good_channel=set(df_graph_gr[df_graph_gr['b']>10]['a'].tolist())
    mask = [i in good_channel for i in graph_data[:,0]]
    graph_data = graph_data[mask]
    print (f"graph data len after filtering: {len(graph_data)}")
    print ('make dataset!!')
    print (dev_feats_with_device_level_feats.shape)
    print (dev_feats.shape)
    print (channel_feats.shape)
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

    X_train,X_test,y_train,y_test = train_test_split(graph_data,graph_data[:,2],test_size = 0.25,random_state = 42)

    return X, X_more_feats,torch.from_numpy(X_train).long(),torch.from_numpy(X_test).long(),split_num

_,X_more_feats,edge_dataset_train,edge_dataset_test,split_num = make_dataset()
X = torch.from_numpy(np.load(f'{date_name}/prepared_data/X_with_feats_selection.npy'))
"""
X:feature matrix for channel-campaign node and device node
X_more_feats:feature matrix for channel-campaign node and device node, the features are un-normalized
edge_dataset: triplet:<channel-campaign node, device node, label> for each edge and the label for the edge
X_train: training set split from edge_dataset
X_test: test set split from edge_dataset
split_num: split num to indicate device features or channel-campaign features in X and X_more_feats
"""

channel_id_max = int(X_more_feats[split_num-1:,20].max().item())
install_carrier_max = int(X_more_feats[:split_num-1,5].max().item())
install_language_max = int(X_more_feats[:split_num-1,6].max().item())
device_brand_max = int(X_more_feats[:split_num-1,7].max().item())
plat_os_max = int(X_more_feats[:split_num-1,9].max().item())

Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print (Device)

model = GAT(edge_dataset_train, edge_dataset_test, split_num, X, X_more_feats, gbm_best_model=None,
                agg='gat',use_graphsage = True,split_bots_normal =False)

model.to(Device)

dset = TensorDataset(edge_dataset_train)
dset_test = TensorDataset(edge_dataset_test)
data_loader = DataLoader(dset, batch_size=256,shuffle = True)
data_loader_test = DataLoader(dset_test, batch_size=256)
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=3e-6)
# model, optimizer = amp.initialize(model, optimizer,opt_level='O1')

train(model, data_loader, data_loader_test, optimizer, 4, thres=0.5, weight=None, agg='gat',  save_name = date_name,multilayer= False,graphsage = True)
