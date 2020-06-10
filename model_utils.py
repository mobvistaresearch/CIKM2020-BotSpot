import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils as utils


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
class Mean_agg(torch.nn.Module):
    '''
    Mean aggregator, following the same logic as in GraphSage
    input x: node features with feature dimensions-${input_dim}  to be averaged and aggregated.
    return: aggregated features of ${hidden_dims} dimensions
    '''

    def __init__(self, input_dim, hidden_dim):
        super(Mean_agg, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h = torch.mean(x, dim=0, keepdim=True)
        h = self.linear(h)
        return h


class Pooling_agg(torch.nn.Module):
    '''
    Pooling aggregator
    input x: node features with feature dimensions-${input_dim}  to be averaged and aggregated.
    return: aggregated features of ${hidden_dims} dimensions
    '''
    def __init__(self, input_dim, hidden_dim):
        super(Pooling_agg, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h, _ = torch.max(x, dim=0, keepdim=True)
        h = self.linear(h)
        return h


class Lstm_agg(torch.nn.Module):
    """
    Aggregation layer using LSTM module
    input_dim:int. dimension for input node
    hidden_dim:int. dimension for hidden output
    Return:
    aggregated features of ${hidden_dims} dimensions
    """
    def __init__(self, input_dim, hidden_dim, num_layer=1):
        super(Lstm_agg, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layer)

    def forward(self, x):
        x = x.unsqueeze(0)
        h = self.lstm(x)
        return h[1][0][0]


class GAT_agg(torch.nn.Module):
    '''
    Graph Attention Aggregation
    linear_dev: linear transformation for device nodes
    linear_channel: linear transformation for channel nodes
    attn: attention layer for weighted combination
    '''

    def __init__(self, input_channel_dim, input_dev_dim, hidden_dim, num_layer=1):
        super(GAT_agg, self).__init__()
        self.linear_dev = nn.Linear(input_dev_dim, hidden_dim)
        self.linear_channel = nn.Linear(input_channel_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, x, h_self):
        x = self.linear_dev(x)
        h_self = self.linear_channel(h_self)
        attn_coef = torch.matmul(self.attn(x), h_self.t().contiguous())
        attn_coef = torch.softmax(attn_coef.t().contiguous(), dim=-1)
        out = torch.matmul(attn_coef, x)
        return out


class Msg_out(torch.nn.Module):
    '''
    nn module for node output after aggregating its neighboring nodes.
    input
    out: aggregated node features after aggregation layer
    x: self node in the lower layer in the graph, with linear transformation and then concat with aggregated node features.
    '''
    def __init__(self, self_dim, hidden_dim):
        super(Msg_out, self).__init__()
        self.linear = nn.Linear(self_dim, hidden_dim)
        self.linear_cluster = nn.Linear(self_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, out, x):
        h = self.linear(x)
        h = self.relu(torch.cat((out, h), dim=1))
        return h


