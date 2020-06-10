from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import  confusion_matrix,accuracy_score
import os
import torch.nn.utils as utils
from model_utils import *

# dataset-1
channel_id_max=1967
install_carrier_max=2768
install_language_max=393
device_brand_max=3220
plat_os_max=13

# uncomment this if try to replicate dataset-2
# dataset-2
# channel_id_max=1544
# install_carrier_max=1918
# install_language_max=317
# device_brand_max=1628
# plat_os_max=33


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Graph_Conv(torch.nn.Module):
    '''
    A tailored graphsage model that aggregate bot node features and normal node features separately and obtain the final node embedding
    using convex combination of the two kind of node features.
    '''

    def __init__(self, dims, num_hop, num_neighbor, edge_index, agg='mean', split_num=-1, reverse=False, data=None,
                 is_left=True, data_more_feats=None, channel_id_emb=1, carrier_emb=1, language_emb=1,
                 device_brand_emb=1, device_name_emb=1, plat_os_emb=1,split_bots_normal = True):
        '''

        :param dims: List[Int], holding input dimensions and hidden dimensions for source node and neighboring node
        :param num_hop: num_hop=1 if only aggregating neighborin nodes for once, if num_hop>1, aggregating nodes revursively.
        :param num_neighbor: maximum number of neighbors sampled when performing aggregation
        :param edge_index: Tuple:: <source node, dst node, label>
        :param agg: agg method, using Mean Aggregation as default
        :param split_num: indicate the rows for which channel node is less than split_num and device node otherwise.
        :param reverse: reverse--> device node to channel node; not reverse: channel node --> device node
        :param data: feature matrix for device node and channel node
        :param is_left: True if channel node else device node
        :param data_more_feats: feature matrix with more sparse high-cardinality categorical features
        :param channel_id_emb: pytorch embedding matrix for channel id
        :param carrier_emb: pytorch embedding matrix for service carrier
        :param language_emb: pytorch embedding matrix for device language
        :param device_brand_emb: pytorch embedding matrix for device brand
        :param device_name_emb: pytorch embedding matrix for device name
        :param plat_os_emb:pytorch embedding matrix for platform os-version combinations
        '''

        if split_num == -1:
            print('forgot to set split_num')
            return
        super(Graph_Conv, self).__init__()
        self.num_hop = num_hop
        self.num_neighbor = num_neighbor
        self.data = data
        self.data_more_feats = data_more_feats
        self.agg = agg
        self.split_num = split_num
        self.left = is_left
        self.module_list = nn.ModuleList([nn.Linear(2, 2)]) # a sentinel layer, not to mean anything
        self.edge_index = edge_index
        self.channel_id_emb = channel_id_emb
        self.carrier_emb = carrier_emb
        self.language_emb = language_emb
        self.device_brand_emb = device_brand_emb
        self.plat_os_emb = plat_os_emb
        self.split_bots_normal = split_bots_normal

        if num_hop <= 1:
            input1_dim, input2_dim, hidden1_dim, hidden2_dim = dims[0], dims[1], dims[2], dims[3]
        elif num_hop == 2:
            input1_dim, input2_dim, hidden1_dim, hidden2_dim, final1_dim, final2_dim = dims[0], dims[1], dims[2], dims[
                3], dims[4], dims[5]
        self.hidden_dims = hidden1_dim + hidden2_dim
        self.hidden1 = hidden1_dim
        self.hidden2 = hidden2_dim
        if not reverse:
            self.input1_dim = input1_dim
            self.input2_dim = input2_dim

        else:
            self.input1_dim = input2_dim
            self.input2_dim = input1_dim
            input1_dim, input2_dim = input2_dim, input1_dim
            hidden1_dim, hidden2_dim = hidden2_dim, hidden1_dim

        if not is_left:
            self.module_list.append(
                nn.ModuleList([Mean_agg(input2_dim, hidden2_dim), Msg_out(input1_dim, hidden1_dim)]))
        elif is_left and agg=='mean':
            self.module_list.append(
                nn.ModuleList([Mean_agg(input2_dim, hidden2_dim), Mean_agg(input2_dim, hidden2_dim),
                               Msg_out(input1_dim, hidden1_dim)]))

        elif is_left and agg=='gat':
            self.module_list.append(
                nn.ModuleList([	GAT_agg(input1_dim,input2_dim, hidden2_dim), GAT_agg(input1_dim,input2_dim, hidden2_dim),
                               Msg_out(input1_dim, hidden1_dim)]))

        if is_left:
            self.attn = nn.Linear(hidden2_dim + 20 + 32, 48)
            self.attn_layer2 = nn.Linear(48, 1)
        self.message_out = Msg_out(input1_dim, hidden1_dim)

    def to_emb(self, arr, left, *models):
        '''
        :param arr: matrix for holding high-cardinality features, without one-hot encoding
        :param left: channel node if left is True else device node
        :param models: a list of embedding matrices to embed each high-cardinality feature to dense embeddings
        :return: 2-d tensor with dense embeddings for all the high-cardinality features.
        '''
        out_arr = []
        arr = arr.long()
        if left:
            # device node sparse features
            dims = [install_carrier_max + 1, install_language_max + 1, device_brand_max + 1, -1,
                    plat_os_max + 1]
        else:
            # channel id for channel node
            dims = [channel_id_max + 1]
        # N = arr.shape[0]
        for i in range(len(dims)):
            if dims[i] == -1: continue
            out_arr.append(models[i](arr[:, i]))
        return torch.cat(out_arr, dim=1)

    def concat_device_feats(self, dev_feats, more_dev_feats):
        '''
        :param dev_feats: normalized device features
        :param more_dev_feats: feature matrix with high-cardinality features
        :return: feature matrix with dense embeddings
        '''
        dev_feats = torch.cat((dev_feats, 1. * more_dev_feats[:, [0]]), dim=1)
        emb_tensor = self.to_emb(more_dev_feats[:, 1:], True, self.carrier_emb, self.language_emb,
                                 self.device_brand_emb, None,
                                 self.plat_os_emb)

        dev_feats = torch.cat((dev_feats, emb_tensor), dim=1)
        return dev_feats

    def concat_channel_feats(self, chan_feats, more_chan_feats):
        '''
        similar to concat_device_feats, to add dense embeddings
        '''
        emb_tensor = self.to_emb(more_chan_feats.view(-1, 1), False, self.channel_id_emb)
        return torch.cat((chan_feats, emb_tensor), dim=1)

    def forward(self, x, nei_vertice):  # x:->List of vertices
        '''

        :param x: List of vertices
        :param nei_vertice:  List of corresponding vertices to x, e.g., an edge (x[0],nei_vertices[0]), (x[1],nei_vertices[1])
        :return: extracted node features for every node in x.
        '''
        out_h = []
        # if nei_vertice not given, then it's a flow of device-->channel as neighboring node is not needed
        if nei_vertice is None:
            for v in x:
                out_h.append(
                    self.induction_hidden_vec(v, self.num_hop, self.edge_index, self.module_list, self.data, self.left))
            # for k in out_h:
                # print ('@@@@ shape:',k.shape)
            nei_emb = torch.cat(out_h, dim=0)
            dev_feats = self.data[x].to(Device)
            more_dev_feats = self.data_more_feats[x][:, 4:10].to(Device)
            dev_feats = self.concat_device_feats(dev_feats, more_dev_feats)
            final_emb = self.message_out(nei_emb, dev_feats)
            return final_emb
        else:
            # if edge_index not a list, then use graphsage without modification
            if not isinstance(self.edge_index, list):
                for v, nei in zip(x, nei_vertice):
                    out_h.append(
                        self.induction_hidden_vec(v, self.num_hop, self.edge_index, self.module_list, self.data,
                                                  self.left, split_bots_normal=False))
                
                nei_emb = torch.cat(out_h, dim=0)
                channel_feats = self.data[x][:,:20].to(Device)
                channel_id_feats = self.data_more_feats[x][:, [20]].to(Device)
                channel_feats = self.concat_channel_feats(channel_feats, channel_id_feats)
                final_emb = self.message_out(nei_emb, channel_feats)
                return final_emb
            else:
                # else, use modified version of graphsage with explicit tailored flow
                h_bots = []
                h_normal = []
                chan_data = self.data[x][:, :20].to(Device)
                chan_id = self.data_more_feats[x][:, 20].to(Device)
                chan_data = self.concat_channel_feats(chan_data, chan_id)

                edge_index_bots = self.edge_index[0] # edges for bot installs
                edge_index_normal = self.edge_index[1] # edges for normal installs
                # extract bot embeddings for every node x
                for v, nei in zip(x, nei_vertice):
                    h_bots.append(
                        self.induction_hidden_vec(v, self.num_hop, edge_index_bots, self.module_list, self.data,
                                                  self.left, split_bots_normal=True, neighbor=nei, bots=True))
                h_bots = torch.cat(h_bots, dim=0)
                # extract normal embeddings for every node x
                for v, nei in zip(x, nei_vertice):
                    h_normal.append(
                        self.induction_hidden_vec(v, self.num_hop, edge_index_normal, self.module_list, self.data,
                                                  self.left, split_bots_normal=True, neighbor=nei, bots=False))

                h_normal = torch.cat(h_normal, dim=0)
                h_bots_out = torch.cat((chan_data, h_bots), dim=1)
                h_normal_out = torch.cat((chan_data, h_normal), dim=1)
                # compute attentional weight for h_bot and h_normal
                alpha_bots = self.attn_layer2(F.relu(self.attn(h_bots_out)))
                alpha_normal = self.attn_layer2(F.relu(self.attn(h_normal_out)))
                attn_coef = F.softmax(torch.cat((alpha_bots, alpha_normal), dim=1), dim=-1)

                h_bots = h_bots.unsqueeze(dim=-1)
                h_normal = h_normal.unsqueeze(dim=-1)

                h_out = torch.cat((h_bots, h_normal), dim=-1)
                attn_coef = attn_coef.unsqueeze(dim=1)
                # print (h_out.shape)
                # print (attn_coef.shape)
                # h_out is the weighted sum of h_bot and h_normal with attentional coefficient alpha.
                h_out = torch.sum(h_out * attn_coef, dim=-1)
                h_out = self.message_out(h_out, chan_data)
                return h_out

    def induction_hidden_vec(self, vertice, num_hop, edge_index, models, embedding_mat,
                             is_left, split_bots_normal=False, neighbor=None,
                             bots=True):  # models:--> [(AGG_MODEL_i,OUT_MODEL_i),(AGG_MODEL_i-1,OUT_MODEL_i-1),.....]
        # print (num_hop)
        '''
        :param vertice: the vertice in the graph as the root node for feature extraction
        :param num_hop: num_hops in graphsage
        :param edge_index: a triplet of <source node, end node, label>
        :param models: embedding matrices for high-cardinality features
        :param embedding_mat:feature matrix for channel nodes and device nodes
        :param is_left: True if flow is channel_node-->device_node, False if flow is device_node --> channel_node
        :param split_bots_normal: True if split bot install and normal install explicitly false otherwise
        :param neighbor: neighboring node of vertice, used if split_bots_normal is True
        :return:
        '''
        min_dim = 20
        try:
            vertice = vertice.item()
        except:
            pass

        if num_hop == 0:
            # basecase, return raw features and dense embeddings
            if vertice < self.split_num - 1:
                # vertice is a device node
                return embedding_mat[[vertice]].float().to(Device), self.data_more_feats[[vertice]][:, 4:10].to(Device)
            else:
                # vertice is a channel-campaign node
                out = embedding_mat[vertice][:min_dim].to(Device)
                cid = self.data_more_feats[vertice, 20].view(1, 1).to(Device)
                return out.float().view(1, -1), cid
        # sample neighbors from edge_index for vertex {vertice}
        neighbors = self.sample_neighbor(edge_index, vertice=vertice, num_samples=self.num_neighbor[num_hop],
                                         left=is_left, neighbor=neighbor)
        # print (neighbors)
        if isinstance(neighbors, int) or len(neighbors)==0:
            if split_bots_normal:
                return torch.zeros(1, self.hidden2).to(Device)
            else:
                print ('return self hidden:',self.hidden_dims)
                return torch.zeros(1, self.hidden_dims).to(Device)
        neighbor_tensor = []
        for n in neighbors:
            neighbor_tensor.append(
                self.induction_hidden_vec(n, num_hop - 1, edge_index, models, embedding_mat, not is_left))

        # neighbor_tensor = torch.cat(neighbor_tensor, dim=0).float()

        neighbor_tensor_without_emb = torch.cat([x[0] for x in neighbor_tensor], dim=0).float()
        if self.left:
            neighbor_tensor_with_emb = torch.cat([x[1] for x in neighbor_tensor], dim=0)
            neighbor_tensor_concat = torch.cat((neighbor_tensor_without_emb, 1. * neighbor_tensor_with_emb[:, [0]]),dim=1)
            emb_tensor = self.to_emb(neighbor_tensor_with_emb[:, 1:], True, self.carrier_emb, self.language_emb,
                                     self.device_brand_emb,
                                     None, self.plat_os_emb)

            neighbor_tensor_concat = torch.cat((neighbor_tensor_concat, emb_tensor), dim=1)
        else:
            neighbor_tensor_with_emb = torch.cat([x[1] for x in neighbor_tensor], dim=0)
            emb_tensor = self.to_emb(neighbor_tensor_with_emb[:, [0]], False, self.channel_id_emb)
            neighbor_tensor_concat = torch.cat((neighbor_tensor_without_emb, emb_tensor), dim=1)
        # if not self.left:
        #     self_tensor = self.induction_hidden_vec(vertice, num_hop - 1, edge_index, models, embedding_mat, is_left)

        if not is_left:
            agg_model, _ = models[num_hop][0], models[num_hop][1]
        elif is_left:
            if bots == True:
                agg_model = models[num_hop][0]
            else:
                agg_model = models[num_hop][1]

        if self.agg != 'gat':
            # use mean aggregation in botspot
            out = agg_model(neighbor_tensor_concat)
            return out
        else:
            # otherwise use GAT in botspot
            self_tensor = self.induction_hidden_vec(vertice, num_hop - 1, edge_index, models, embedding_mat, not is_left)
            cid = self_tensor[1][0].long()
            channel_feats = torch.cat((self_tensor[0],self.channel_id_emb(cid).to(Device)),dim=1)
            # print(channel_feats.shape)
            out = agg_model(neighbor_tensor_concat, channel_feats)
            return out

    def sample_neighbor(self, edges, num_samples=20, vertice=0, left=True, neighbor=None):
        try:
            vertice = vertice.item()
        except:
            pass

        try:
            neighbor = neighbor.item()
        except:
            pass

        if left and self.split_bots_normal:
            assert neighbor is not None
            nei = edges[edges[:, 0] == vertice][:, 1]
            if len(nei) == 0:
                return -1
            if len(nei) == 1 and neighbor in nei: return -1

            if len(nei) <= num_samples:
                try:
                    # for channel node, it is important not to include the neighbor node itself.
                    nei = np.delete(nei, np.where(nei == neighbor))  # endpoint of the edge has to be deleted to avoid information leakage
                except:
                    pass
                return nei

            vertices = np.random.choice(nei, size=num_samples)
            try:
                vertices = np.delete(vertices, np.where(vertices == neighbor))
            except:
                pass
            # print ('channel neighbor num:',len(nei))
            return vertices
        else:
            if not left:
                direct_neighbor = edges[edges[:, 1] == vertice][:, 0]
            else:
                direct_neighbor = edges[edges[:, 0] == vertice][:, 1]
            # print ('device neighbor num:',len(direct_neighbor))
            if len(direct_neighbor) == 0:
                print ('device neighbor num:',vertice)
                return -1
            if len(direct_neighbor) <= num_samples:
                return direct_neighbor
            else:
                vertices = np.random.choice(direct_neighbor, size=num_samples)
                return vertices


class BotSpot(torch.nn.Module):
    def __init__(self,edge_index_train,edge_index_test, split_num, data, data_more_feats=None, gbm_models=None,
                 gbm_best_model=None, agg='mean',use_graphsage = False,split_bots_normal = True):
        super(BotSpot, self).__init__()
        leaf_dim = 64
        self.num_gbm_trees = 200
        try:
            edge_index_train = edge_index_train.numpy()
            edge_index_test = edge_index_test.numpy()
        except:
            pass
        self.channel_id_emb = nn.Embedding(channel_id_max + 1, 32)
        self.carrier_emb = nn.Embedding(install_carrier_max + 1, 32)
        self.language_emb = nn.Embedding(install_language_max + 1, 32)
        self.device_brand_emb = nn.Embedding(device_brand_max + 1, 32)
        self.device_name_emb = nn.Linear(1 + 1, 12)
        self.plat_os_emb = nn.Embedding(plat_os_max + 1, 8)
        self.gbm_best_model = gbm_best_model
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        # construct embedding matrices for leaf embedding
        if gbm_best_model is not None:
            self.leaf_emb_models = nn.ModuleList()
            for n in range(gbm_best_model.n_estimators):
                self.leaf_emb_models.append(nn.Embedding(31, leaf_dim))

        edge_index = np.vstack((edge_index_train, edge_index_test))
        edge_index_bots = edge_index_train[edge_index_train[:,2]==1]
        edge_index_normal = edge_index_train[edge_index_train[:,2]==0]
        if use_graphsage:
            channel_to_device_flow = edge_index
        else:
            channel_to_device_flow = [edge_index_bots, edge_index_normal]
        # dims: [local_dims,remote_dims,local_hidden_dims,remote_hidden_dims]
        # graph_model_rev is the

        self.graph_model = Graph_Conv(dims=[20 + 32, 22 + 3 * 32 + 1 + 8, 32, 64], num_hop=1, num_neighbor=[0, 30],
                                      edge_index=channel_to_device_flow, split_num=split_num,
                                      reverse=False, data=data,
                                      is_left=True, agg=agg, data_more_feats=data_more_feats,
                                      channel_id_emb=self.channel_id_emb,
                                      carrier_emb=self.carrier_emb, language_emb=self.language_emb,
                                      device_brand_emb=self.device_brand_emb,
                                      device_name_emb=self.device_name_emb, plat_os_emb=self.plat_os_emb,split_bots_normal=split_bots_normal)
        self.device_linear = nn.Linear(127, 64)
        self.device_dropout = nn.Dropout(0.4)
        self.device_norm  = nn.BatchNorm1d(64)
        self.linear_edge = nn.Linear(32 + 64 + 64, 64)
        self.edge_dropout = nn.Dropout(0.4)
        self.edge_norm = nn.BatchNorm1d(64)
        # self.left_linear = nn.Linear(16 + 28, 24)
        # self.linear_top = nn.Linear(24+24+24, 48)
        self.gbm_models = gbm_models
        self.data = data
        self.data_more_feats = data_more_feats

        if gbm_models is None and gbm_best_model is None:
            self.linear_local1 = nn.Linear(64, 48)
            self.linear_local2 = nn.Linear(48,1)
        elif gbm_models is not None:
            self.linear_top2 = nn.Linear(64, 1)
            self.linear_top3 = nn.Linear(1 + 1, 1, bias=False)  # len(gbm_models)
        elif gbm_best_model is not None:
            # 50 trees
            self.leaf_linear1 = nn.Linear(leaf_dim,48)
            # self.leaf_norm1 = nn.BatchNorm1d(48)
            # self.leaf_linear2 = nn.Linear(48, 32)
            # self.leaf_norm2 = nn.BatchNorm1d(32)

            self.linear_top2 = nn.Linear(64, 32)
            self.linear_top_norm = nn.BatchNorm1d(32)
            self.final_pred = nn.Linear(32+48,1)
            # 100 trees
            # self.leaf_linear1 = nn.Linear(leaf_dim*100,1024)
            # self.leaf_linear2 = nn.Linear(1024, 384)
            # self.linear_top2 = nn.Linear(128, 96)
            # self.final_pred = nn.Linear(96+384,256)
            # self.final_pred_out = nn.Linear(256,1)

            # 300 trees
            # self.leaf_linear1 = nn.Linear(leaf_dim * 300, 128)
            # self.leaf_linear2 = nn.Linear(128, 96)
            # self.leaf_linear3 = nn.Linear(96, 64)
            # self.linear_top2 = nn.Linear(128, 96)
            # self.final_pred = nn.Linear(96 + 64, 128)
            # self.final_pred_out = nn.Linear(128, 1)


    def to_emb(self, arr, left, *models):
        '''
        :param arr: matrix for holding high-cardinality features, without one-hot encoding

        :param left: channel node if left is True else device node
        :param models: a list of embedding matrices to embed each high-cardinality feature to dense embeddings
        :return: 2-d tensor with dense embeddings for all the high-cardinality features.
        '''
        out_arr = []
        arr = arr.long()
        if left:
            # device node sparse features
            dims = [install_carrier_max + 1, install_language_max + 1, device_brand_max + 1, -1,
                    plat_os_max + 1]
        else:
            # channel id for channel node
            dims = [channel_id_max + 1]
        # N = arr.shape[0]
        for i in range(len(dims)):
            if dims[i] == -1: continue
            out_arr.append(models[i](arr[:, i]))
        return torch.cat(out_arr, dim=1)

    def concat_device_feats(self, dev_feats, more_dev_feats):
        '''
        a method to embed high-cardinality features for device
        :param dev_feats: normalized device features

        :param more_dev_feats: feature matrix with high-cardinality features

        :return: feature matrix with dense embeddings

        '''
        dev_feats = torch.cat((dev_feats, 1. * more_dev_feats[:, [0]]), dim=1)
        emb_tensor = self.to_emb(more_dev_feats[:, 1:], True, self.carrier_emb, self.language_emb,
                                 self.device_brand_emb, None,
                                 self.plat_os_emb)

        dev_feats = torch.cat((dev_feats, emb_tensor), dim=1)
        return dev_feats


    def forward(self, edges):
        left_v = edges[:, 0]
        right_v = edges[:, 1]
        labels = edges[:, 2]

        dev_feats = self.data[right_v].to(Device)
        more_dev_feats = self.data_more_feats[right_v][:, 4:10].to(Device)
        right_feats = self.concat_device_feats(dev_feats, more_dev_feats)  # embed high cardinality features for device
        right_feats = F.relu(self.device_linear(right_feats)) # right feats is device feature without graph convolution
        left_feats = self.graph_model(left_v, right_v)  # extract features from channel-campaign node using tailored propagation method
        h_edge = F.relu(self.linear_edge(torch.cat((left_feats,right_feats),dim=1)))
        # if no gbm models, use botspot-local only
        if self.gbm_models is None and self.gbm_best_model is None:  # if no gbm models, it's botspot-local
            h = torch.sigmoid(self.linear_local2(F.relu(self.linear_local1(h_edge))))
            return h, labels
        elif self.gbm_models is not None: # deprecated
            # use gbm_mdoels for probability of a sample as meta information
            gbm_outputs = self.pred_proba_from_light_gbm(left_v, right_v)
            h_edge = torch.cat((h_edge, gbm_outputs), dim=1)
            return self.linear_top3(h_edge),labels

        elif self.gbm_best_model is not None:
            # extract leaf node embedding and
            # get leaf index from gbm model and fed into leaf embedding matrix for model training
            pred_leaf = self.get_leaf_from_light_gbm(left_v,right_v)
            pred_leaf_h = F.relu(self.leaf_linear1(pred_leaf))  #100trees

            h_edge = F.relu(self.linear_top2(h_edge))
            out = torch.sigmoid(self.final_pred(torch.cat((h_edge,pred_leaf_h),dim=1)))
            # print ('forward:',out)
            return out,labels

    def pred_proba_from_light_gbm(self, left_vertices, right_vertices):
        # deprecated, not used in this project
        assert self.data_more_feats is not None
        chan_data = self.data_more_feats[left_vertices][:, :21]
        dev_data = self.data_more_feats[right_vertices]
        edge_data = torch.cat((chan_data, dev_data), dim=1)
        edge_data = edge_data.numpy()
        D = len(self.gbm_models)
        out = np.zeros((len(left_vertices), D))

        for idx, gbm in enumerate(self.gbm_models):
            out[:, idx] = gbm.predict_proba(edge_data)[:, 1]

        return torch.mean(torch.from_numpy(out).float(), dim=1, keepdim=True)

    def get_leaf_from_light_gbm(self, left_vertices, right_vertices):
        # get leaf indices from gbm model and embed into dense matrix
        output_leaf_emb = []
        chan_data = self.data_more_feats[left_vertices][:, :21]
        dev_data = self.data_more_feats[right_vertices]
        edge_data = torch.cat((chan_data, dev_data), dim=1) # edge feature is the concatenation of channel_node and device_node
        edge_data = edge_data.numpy()
        # N = len(left_vertices)
        pred_leaf = self.gbm_best_model.predict_proba(edge_data, pred_leaf=True)
        pred_leaf = torch.from_numpy(pred_leaf).long().to(Device)

        for i in range(pred_leaf.shape[1]):
            # print (self.leaf_emb_models[i](pred_leaf[:, i]).shape)
            output_leaf_emb.append(self.leaf_emb_models[i](pred_leaf[:, i]).unsqueeze(1))
            # ret = torch.cat(output_leaf_emb, dim=1).to(Device)  # leaf node concatenation
        ret=torch.cat(output_leaf_emb, dim=1).mean(axis=1).to(Device) # leaf node mean pooling
        return ret




