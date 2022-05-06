# from __future__ import division
# from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from utils import generator, accuracy
import config
from models import *
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import pdb

# Training settings
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=7, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01, #0.015
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--model', type=str, default='GCN',
                    help='model type')
parser.add_argument('--task', type=str, default='sex_predict', 
                    help='task type')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#   random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def sex_predict_criterion(pred, target):
    BCE = nn.BCELoss()
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    return BCE(pred, target) + 0.5*L1(pred, target)

def age_predict_criterion(pred, target):
    BCE = nn.BCELoss()
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    return MSE(pred, target) 


def brain_batch_train(model, task, criterion, adj, label, feature, optimizer):
    model.train()
    optimizer.zero_grad()
    train_output = model(feature, adj, task)
    loss_train = criterion(train_output, label)
    acc_train = accuracy(train_output, label)
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train, len(train_output)


def brain_batch_test(model, task, criterion, adj, label, feature, predp, gt):
    model.eval()
    with torch.no_grad():
        test_output = model(feature, adj, task)
        loss_test = criterion(test_output, label)
        acc_test = accuracy(test_output, label)
        predp.extend(test_output)
        gt.extend(label)
    return loss_test, acc_test, len(test_output), predp, gt


def brain_train():
    train_idx_fold = np.load(config.train_idx_fold, allow_pickle=True)
    test_idx_fold = np.load(config.test_idx_fold, allow_pickle=True)
    best_accall = []
    best_aucall = []
    for i in range(3,4):
        best_acc = 0
        best_auc = 0
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        task = args.task
        #   Task Type
        if task == 'sex_predict':
            criterion = sex_predict_criterion
        elif task == 'age_predict':
            criterion = age_predict_criterion
        #   Model Type
        if args.model == 'GCN':
            net = GCN(160, 80, 1, args.dropout)
        if args.model == 'MLP':
            net = MLP(6400 * 4, 1, args.dropout)
        elif args.model == 'GCN_specific_L_W':
            net = GCN_specific_L_W(160, 80, 1, args.dropout)
        elif args.model == 'GCN_general_L_W':
            net = GCN_general_L_W(160, 80, 1, args.dropout)
        optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)
        print('==================================== fold:{} ============================================'.format(i+1))
        loss_train_list, acc_train_list, len_train_list = [], [], []
        loss_test_list, acc_test_list, len_test_list = [], [], []
        train_generator, test_generator = generator(train_idx_fold[i], test_idx_fold[i], args.batch_size)
        for epoch in range(args.epochs):
            st = time.time()
            predpall, gtall = [], []
            for adj_train, label_train, feature_train, train_idx in train_generator:
                loss_train, acc_train, len_train = brain_batch_train(net, task, criterion, adj_train, label_train, feature_train, optimizer)
                loss_train_list.append(loss_train*len_train)
                acc_train_list.append(acc_train*len_train)
                len_train_list.append(len_train)
            for adj_test, label_test, feature_test, _ in test_generator:
                loss_test, acc_test, len_test, predpall, gtall = brain_batch_test(net, task, criterion, adj_test, label_test, feature_test, predpall, gtall)
                loss_test_list.append(loss_test*len_test)
                acc_test_list.append(acc_test*len_test)
                len_test_list.append(len_test)
            predpall = np.array([p.item() for p in predpall])
            gtall = np.array([g.item() for g in gtall])
            auc = roc_auc_score(gtall, predpall)
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(sum(loss_train_list)/sum(len_train_list)),
                  'acc_train: {:.4f}'.format(sum(acc_train_list)/sum(len_train_list)),
                  'loss_test: {:.4f}'.format(sum(loss_test_list)/sum(len_test_list)),
                  'acc_test: {:.4f}'.format(sum(acc_test_list)/sum(len_test_list)),
                  'auc_test: {:.4f}'.format(auc),
                  'time: {:.4f}s'.format(time.time() - st))
            acc_test = sum(acc_test_list)/sum(len_test_list)
            if acc_test > best_acc:
                best_acc = acc_test
            if auc > best_auc:
                best_auc = auc
        best_accall.append(best_acc)
        best_aucall.append(best_auc)
    d = {'best_accall': best_accall, 'best_aucall': best_aucall}
    df = pd.DataFrame(data=d)
    df.to_csv('result.csv', index=False)
    print('best accall is:', best_accall)
    print('best aucall is:', best_aucall)
    return best_accall, best_aucall





if __name__ == '__main__':
    best_accall, best_aucall = brain_train()




