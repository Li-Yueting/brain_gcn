import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import os
import pdb
#os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
# from torch_geometric.nn import global_mean_pool, GATConv


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.batch1 = nn.BatchNorm1d(input_dim)
        self.input_fc = nn.Linear(input_dim, input_dim // 20)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.output_fc = nn.Linear(input_dim // 20, 1)
        self.LogSoftmax = nn.LogSoftmax()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, adj, task):
        # x = [batch size, height, width]
        batch_size = adj.shape[0]
        x1 = adj.view(batch_size, -1)
        l1 = self.input_fc(x1)
        l2 = self.drop(self.activation(l1))
        l2 = self.output_fc(l2)
        if task=='sex_predict':
            pred = self.Sigmoid(l2)
        elif task=='age_predict':
            pred=l2
        return pred.reshape(pred.shape[0])


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        # self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(input_dim // 2 * hidden_dim // 2)
        self.fc1 = nn.Linear(input_dim // 2 * hidden_dim // 2, 128)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, adj, task):
        x = F.selu(self.gc1(x, adj))
        x = self.drop(x)
        l1 = self.avgpool(x)  # F.adaptive_avg_pool2d(x, [40, 32])#self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = self.fc2(l2)
        if task=='sex_predict':
            pred = self.Sigmoid(l2)
        elif task=='age_predict':
            pred=l2
        return pred.reshape(pred.shape[0])


class GCN_specific_L_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_specific_L_W, self).__init__()
        c_num = (input_dim // 2) * (input_dim // 2) * 2
        self.learn_weight_layer = nn.Sequential(
            nn.BatchNorm1d(c_num),
            nn.Linear(c_num, c_num // 20),
            #nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(c_num // 20, input_dim // 2)
        )
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(input_dim//2 * hidden_dim//2)
        self.fc1 = nn.Linear(input_dim//2 * hidden_dim//2, 128)
        self.drop = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, joint, task):
        FC = joint[:, 0:80, 0:80]
        SC = joint[:, 80:160, 80:160]
        # FC: b * 80 * 80
        # SC: b * 80 * 80
        b, w, h = FC.shape
        weight = self.learn_weight_layer(torch.cat([FC, SC], dim=0).view(b, -1))  ### -> b, 80
        print('the weight is ... ', weight)
        weight = torch.sigmoid(weight)
        mapcat = torch.zeros_like(FC)
        for i in range(w):
            mapcat[:, i, i] = weight[:, i]
        mapcatsub1 = torch.cat([FC, mapcat], 1)
        mapcatsub2 = torch.cat([mapcat, SC], 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)
        x = F.selu(self.gc1(x, adj))  
        x = self.drop(x)
        l1 = self.avgpool(x)  
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = self.fc2(l2)
        if task=='sex_predict':
            pred = self.Sigmoid(l2)
        elif task=='age_predict':
            pred=l2
        return pred.reshape(pred.shape[0])


class GCN_general_L_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_general_L_W, self).__init__()
        c_num = (input_dim // 2) * (input_dim // 2) * 2
        self.learn_weight_layer = nn.Sequential(
            nn.Linear(1, 80),
            #nn.Dropout(dropout),
            nn.ReLU(),
            #nn.Linear(80, input_dim // 2)
        )
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch_gcn = nn.BatchNorm1d(input_dim)
        self.batch = nn.BatchNorm1d(input_dim // 2 * hidden_dim // 2)
        self.fc1 = nn.Linear(input_dim // 2 * hidden_dim // 2, 128)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.Sigmoid = nn.Sigmoid()


    def forward(self, x, joint, task):
        #### FC: b * 80 * 80
        #### SC: b * 80 * 80
        FC = joint[:, 0:80, 0:80]
        SC = joint[:, 80:160, 80:160]
        b, w, h = FC.shape
        weight = self.learn_weight_layer(torch.ones(1))  ### -> b, 80
        weight = torch.sigmoid(weight)
        mapcat = torch.zeros_like(FC)
        for i in range(b):
            #pdb.set_trace()
            mapcat[i, :, :] = torch.diag(weight)
        mapcatsub1 = torch.cat([FC, mapcat], 1)
        mapcatsub2 = torch.cat([mapcat, SC], 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)
        x = F.selu(self.gc1(x, adj))
        x = self.drop(x)
        l1 = self.avgpool(x) 
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = self.fc2(l2)
        if task=='sex_predict':
            pred = self.Sigmoid(l2)
        elif task=='age_predict':
            pred=l2
        return pred.reshape(pred.shape[0])
