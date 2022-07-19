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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Training settings
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=7, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001, #0.015
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--model', type=str, default='GCN_general_L_W',
                    help='model type')
parser.add_argument('--task', type=str, default='age_predict',
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
    return BCE(pred, target) #+ 0.5*L1(pred, target)


def age_predict_criterion(pred, target):
    BCE = nn.BCELoss()
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    return MSE(pred, target) 


def brain_batch_train(model, task, criterion, adj, label, feature, optimizer, scanner_machine):
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    optimizer.zero_grad()
    B = adj.shape[0]
    if args.model == 'MLP':
        inp = feature.view(B, -1).cuda()#torch.cat([feature.view(B, -1), adj[:, :80, :80].contiguous().view(B, -1), adj[:, 80:, 80:].contiguous().view(B, -1)], dim=1).cuda()
        train_output = model(inp, scanner_machine.cuda(), task)
    elif args.model == 'MLP_160':
        fea = []
        for i in range(B):
            fea.append(np.concatenate([np.diag(feature[i, :80]), np.diag(feature[i, 80:])]))
        feature = np.stack(fea)
        feature = torch.tensor(feature)
        inp = feature.view(B, -1).cuda()  # torch.cat([feature.view(B, -1), adj[:, :80, :80].contiguous().view(B, -1), adj[:, 80:, 80:].contiguous().view(B, -1)], dim=1).cuda()
        train_output = model(inp, scanner_machine.cuda(), task)
    else:
        train_output = model(feature.cuda(), adj.cuda(), task, scanner_machine.cuda())
    loss_train = criterion(train_output, label.cuda())
    acc_train = accuracy(train_output.cpu(), label)
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train, len(train_output)


def brain_batch_test(model, task, criterion, adj, label, feature, scanner_machine, predp, gt):
    model.eval()
    B = adj.shape[0]
    with torch.no_grad():
        if args.model == 'MLP':
            inp = feature.view(B, -1).cuda()#torch.cat([feature.view(B, -1), adj[:, :80, :80].contiguous().view(B, -1), adj[:, 80:, 80:].contiguous().view(B, -1)], dim=1).cuda()
            test_output = model(inp, scanner_machine.cuda(), task)
        elif args.model == 'MLP_160':
            fea = []
            for i in range(B):
                fea.append(np.concatenate([np.diag(feature[i, :80]), np.diag(feature[i, 80:])]))
            feature = np.stack(fea)
            feature = torch.tensor(feature)
            inp = feature.view(B, -1).cuda()  # torch.cat([feature.view(B, -1), adj[:, :80, :80].contiguous().view(B, -1), adj[:, 80:, 80:].contiguous().view(B, -1)], dim=1).cuda()
            test_output = model(inp, scanner_machine.cuda(), task)
        else:
            test_output = model(feature.cuda(), adj.cuda(), task, scanner_machine.cuda())
        loss_test = criterion(test_output, label.cuda())
        acc_test = accuracy(test_output.cpu(), label)
        predp.extend(test_output.cpu())
        gt.extend(label)
    return loss_test, acc_test, len(test_output), predp, gt


def brain_train():
    train_idx_fold = np.load(config.train_idx_fold, allow_pickle=True)
    test_idx_fold = np.load(config.test_idx_fold, allow_pickle=True)
    best_acc_5fold, best_acc_auc_5fold, best_auc_5fold, best_auc_acc_5fold = [], [], [], []
    best_mae_5fold, best_mae_pearson_5fold, best_pearson_5fold, best_pearson_mae_5fold = [], [], [], []
    for i in range(5):
        # if args.cuda:
        #     torch.cuda.manual_seed(args.seed)
        task = args.task
        best_acc, best_acc_auc = 0, 0
        best_auc, best_auc_acc = 0, 0
        best_mae, best_mae_pearson = 1000, 0
        best_pearson, best_pearson_mae = 0, 1000
        #   Task Type
        if task == 'sex_predict':
            criterion = sex_predict_criterion
        elif task == 'age_predict':
            criterion = age_predict_criterion
        #   Model Type
        if args.model == 'GCN':
            net = GCN(160, 80, 1, args.dropout)
        elif args.model == 'MLP':
            net = MLP(6400 * 2, 1, args.dropout)
        elif args.model == 'MLP_160':
            net = MLP(80 * 2, 1, args.dropout)
        elif args.model == 'GCN_specific_L_W':
            net = GCN_specific_L_W(160, 80, 1, args.dropout)
        elif args.model == 'GCN_general_L_W':
            net = GCN_general_L_W(80, 40, 1, args.dropout)
        elif args.model == 'GCN_dense_L_W':
            net = GCN_dense_L_W(80, 40, 1, args.dropout)
        elif args.model =='GCN_MV_GCN' :
            net = GCN_MV_GCN(80, 80, 40, 1, args.dropout)
        net.cuda()
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
        print('==================================== fold:{} ============================================'.format(i+1))
        train_generator, test_generator = generator(train_idx_fold[i], test_idx_fold[i], args.batch_size, task)
        for epoch in range(args.epochs):
            st = time.time()
            loss_train_list, acc_train_list, len_train_list = [], [], []
            loss_test_list, acc_test_list, len_test_list = [], [], []
            predpall, gtall = [], []
            for adj_train, label_train, feature_train, train_m, _ in train_generator:
                loss_train, acc_train, len_train = brain_batch_train(net, task, criterion, adj_train, label_train, feature_train, optimizer, train_m)
                loss_train_list.append(loss_train*len_train)
                acc_train_list.append(acc_train*len_train)
                len_train_list.append(len_train)
            for adj_test, label_test, feature_test, test_m, _ in test_generator:
                loss_test, acc_test, len_test, predpall, gtall = brain_batch_test(net, task, criterion, adj_test, label_test, feature_test, test_m, predpall, gtall)
                loss_test_list.append(loss_test*len_test)
                acc_test_list.append(acc_test*len_test)
                len_test_list.append(len_test)
            predpall = np.array([p.item() for p in predpall])
            gtall = np.array([g.item() for g in gtall])
            if task == 'sex_predict':
                acc = accuracy_score(predpall > 0.5, gtall)
                auc = roc_auc_score(gtall, predpall)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(sum(loss_train_list) / sum(len_train_list)),
                      'acc_train: {:.4f}'.format(sum(acc_train_list) / sum(len_train_list)),
                      'loss_test: {:.4f}'.format(sum(loss_test_list) / sum(len_test_list)),
                      'acc_test: {:.4f}'.format(acc),
                      'auc_test: {:.4f}'.format(auc),
                      'time: {:.4f}s'.format(time.time() - st))
                if acc > best_acc:
                    best_acc = acc
                    best_acc_auc = auc
                    torch.save(net.state_dict(), './best_models/' + args.model + '_net_acc_' + task + '_' + str(i) + '.pth')
                if auc > best_auc:
                    best_auc = auc
                    best_auc_acc = acc
                    torch.save(net.state_dict(), './best_models/' + args.model + '_net_auc_' + task + '_' + str(i) + '.pth')
            if task == 'age_predict':
                mae = mean_absolute_error(gtall, predpall)
                pearson = pearsonr(gtall, predpall)[0]
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(sum(loss_train_list) / sum(len_train_list)),
                      'loss_test: {:.4f}'.format(sum(loss_test_list) / sum(len_test_list)),
                      'mea_test: {:.4f}'.format(mae),
                      'pearson_test: {:.4f}'.format(pearson),
                      'time: {:.4f}s'.format(time.time() - st))
                if mae < best_mae:
                    best_mae = mae
                    best_mae_pearson = pearson
                    torch.save(net.state_dict(),
                               './best_models/' + args.model + '_net_mae_' + task + '_' + str(i) + '.pth')
                if pearson > best_pearson:
                    best_pearson = pearson
                    best_pearson_mae = mae
                    torch.save(net.state_dict(),
                               './best_models/' + args.model + '_net_pearson_' + task + '_' + str(i) + '.pth')
        best_acc_5fold.append(best_acc)
        best_auc_5fold.append(best_auc)
        best_acc_auc_5fold.append(best_acc_auc)
        best_auc_acc_5fold.append(best_auc_acc)
        best_mae_5fold.append(best_mae)
        best_mae_pearson_5fold.append(best_mae_pearson)
        best_pearson_5fold.append(best_pearson)
        best_pearson_mae_5fold.append(best_pearson_mae)
    if task == 'sex_predict':
        d = {'best_acc_5fold': best_acc_5fold, 'best_acc_auc_5fold': best_acc_auc_5fold, 'best_auc_5fold': best_auc_5fold, 'best_auc_acc_5fold': best_auc_acc_5fold}
        df = pd.DataFrame(data=d)
        df.to_csv('./results/result_' + args.model + task + '_.csv', index=False)
        print('best best_acc_5fold is:', best_acc_5fold, 'mean acc of them is:', sum(best_acc_5fold)/len(best_acc_5fold))
        print('best best_auc_5fold is:', best_auc_5fold, 'mean auc of them is:', sum(best_auc_5fold)/len(best_auc_5fold))
        return best_acc_5fold, best_auc_5fold
    if task == 'age_predict':
        d = {'best_mae_5fold': best_mae_5fold, 'best_mae_pearson_5fold': best_mae_pearson_5fold, 'best_pearson_5fold': best_pearson_5fold, 'best_pearson_mae_5fold': best_pearson_mae_5fold}
        df = pd.DataFrame(data=d)
        df.to_csv('./results/result_' + args.model + task + '_.csv', index=False)
        print('best_mae_5fold is:', best_mae_5fold, 'mean mae of them is:', sum(best_mae_5fold)/len(best_mae_5fold),\
        "mean pearson of them is:", sum(best_mae_pearson_5fold)/len(best_mae_pearson_5fold))
        print('best_pearson_5fold is:', best_pearson_5fold, 'mean pearson of them is:', sum(best_pearson_5fold)/len(best_pearson_5fold),\
        "mean mae of them is:", sum(best_pearson_mae_5fold)/len(best_pearson_mae_5fold))
        

if __name__ == '__main__':
    print(args.model + '_' + args.task)
    brain_train()

    





