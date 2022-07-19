import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import os
import pdb


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, input_dim // 4)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.SELU()#ReLU()
        self.output_fc = nn.Linear(input_dim // 4 + 1, 1)
        self.LogSoftmax = nn.LogSoftmax()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, wm, task):
        # x/adj size: [batch size, height, width]
        batch_size = x.shape[0]
        x1 = x.view(batch_size, -1)
        l1 = self.input_fc(x1)
        l2 = self.drop(self.activation(l1))
        l2 = torch.cat([l2, wm], dim=1)
        l2 = self.output_fc(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
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
        self.fc2 = nn.Linear(128 + 1, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, adj, task, wm):
        x = F.selu(self.gc1(x, adj))
        x = self.drop(x)
        l1 = self.avgpool(x)  # F.adaptive_avg_pool2d(x, [40, 32])#self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = torch.cat([l2, wm], dim=1)
        l2 = self.fc2(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
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
        self.fc2 = nn.Linear(129, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, joint, task, scanner_machine):
        FC = joint[:, 0:80, 0:80]
        SC = joint[:, 80:160, 80:160]
        # FC: b * 80 * 80
        # SC: b * 80 * 80
        b, w, h = FC.shape
        weight = self.learn_weight_layer(torch.cat([FC, SC], dim=0).view(b, -1))  ### -> b, 80
        weight = torch.sigmoid(weight)
        # print('weight', weight)
        mapcat = torch.zeros_like(FC)
        for i in range(w):
            mapcat[:, i, i] = weight[:, i]
        mapcatsub1 = torch.cat((FC, mapcat), 1)
        mapcatsub2 = torch.cat((mapcat, SC), 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)
        # print(x.shape)
        x = F.selu(self.gc1(x, adj))  
        x = self.drop(x)
        l1 = self.avgpool(x)  
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = torch.cat([l2, scanner_machine], dim=1)
        l2 = self.fc2(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])


class GCN_general_L_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_general_L_W, self).__init__()
        self.learn_weight_layer = nn.Sequential(
            nn.Linear(1, 80),
            # nn.Dropout(dropout),
            # nn.ReLU(),
            # nn.Linear(80, 80)
        )
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(3200//2)
        self.fc1 = nn.Linear(3200//2, 128)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(129, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def get_learning_weight(self):
        return self.learned_weight

    def forward(self, x, joint, task, scanner_machine):
        FC = joint[:, 0:80, 0:80]
        SC = joint[:, 80:160, 80:160]
        b, w, h = FC.shape
        weight = self.learn_weight_layer(torch.ones(1).cuda())  ### -> b, 80
        weight = torch.sigmoid(weight)
        self.learned_weight = weight
        # print('weights', weight)
        mapcat = torch.zeros_like(FC)
        for i in range(b):
            mapcat[i, :, :] = torch.diag(weight)
        mapcatsub1 = torch.cat([FC, mapcat], 1)
        mapcatsub2 = torch.cat([mapcat, SC], 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)
        x = F.selu(self.gc1(x, adj)) 
        x = self.dropout(x)
        l1 = self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.relu(self.fc1(l1))
        l2 = self.dropout(l2)
        l2 = torch.cat((l2, scanner_machine), dim=1)
        l2 = self.fc2(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])

class GCN_dense_L_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_dense_L_W, self).__init__()
        self.learn_weight_layer = nn.Sequential(
            nn.Linear(1, 6400),
            # nn.Dropout(dropout),
            # nn.ReLU(),
            # nn.Linear(80, 80)
        )
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(3200//2)
        self.fc1 = nn.Linear(3200//2, 128)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(129, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def get_learning_weight(self):
        return self.learned_weight

    def forward(self, x, joint, task, scanner_machine):
        FC = joint[:, 0:80, 0:80]
        SC = joint[:, 80:160, 80:160]
        b, w, h = FC.shape
        weight = self.learn_weight_layer(torch.ones(1).cuda())  ### -> b, 80
        weight = torch.sigmoid(weight)
        weight = weight.reshape(80, 80)
        # self.learned_weight = weight
        # print('weights', weight)
        mapcat = torch.zeros_like(FC)
        for i in range(b):
            mapcat[i, :, :] = weight
        mapcatsub1 = torch.cat([FC, mapcat], 1)
        mapcatsub2 = torch.cat([mapcat, SC], 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)
        x = F.selu(self.gc1(x, adj)) 
        x = self.dropout(x)
        l1 = self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.relu(self.fc1(l1))
        l2 = self.dropout(l2)
        l2 = torch.cat((l2, scanner_machine), dim=1)
        l2 = self.fc2(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])


class GCN_MV_GCN(nn.Module):
    def __init__(self, Node_num, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_MV_GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(input_dim, hidden_dim)
        # self.gc2 = GraphConvolution(hidden_dim, hidden_dim)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(Node_num * hidden_dim // 2)
        self.batch_gcn = nn.BatchNorm1d(Node_num)
        self.fc1 = nn.Linear(Node_num * hidden_dim // 2, 128)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        # self.batch2 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 64)

        self.fc2 = nn.Linear(128 + 1, output_dim)
        self.Sigmoid = nn.Sigmoid()
        """
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.linear = nn.Linear(nfeat, 1)
        self.weight = nn.Parameter(torch.randn(1, nfeat))
        self.bias = nn.Parameter(torch.randn(1))
        """


    def forward(self, feature, adj, task, mw):
        #(self, feature, adj, mw, device=None):
        #### FC: b * 80 * 80
        #### SC: b * 80 * 80
        FC_F = feature[:, :80, :]
        SC_F = feature[:, 80:, :]
        FC = adj[:, :80, :80]
        SC = adj[:, 80:160, 80:160]

        # feature = self.learn_feature_layer(feature)
        x_F = F.selu(self.batch_gcn(self.gc1(FC_F, FC)))  # input_dim, hidden_dim
        x_S = F.selu(self.batch_gcn(self.gc2(SC_F, SC)))  # input_dim, hidden_dim
        x = torch.cat([x_F, x_S], dim=1)
        x = self.drop(x)
        l1 = self.avgpool(x)  # input_dim//2, hidden_dim//2
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = torch.cat([l2, mw], dim=1)
        l2 = self.fc2(l2)

        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])
