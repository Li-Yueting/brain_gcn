import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

import config

import pdb
import logging
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generator(train_idx, test_idx, batch_size):
    label = pd.read_csv(config.label_file)['sex'].values
    # TODO:
    g_list = np.load(config.JOINT_path, allow_pickle=True)
    f_list = []
    for _ in range(len(g_list)):
        f_list.append(np.eye(160))
    train_g = torch.FloatTensor(np.array(g_list)[train_idx])
    train_label = torch.FloatTensor(np.array(label)[train_idx])
    train_f = torch.FloatTensor(np.array(f_list)[train_idx])
    train_idx = torch.FloatTensor(np.array(train_idx))
    train_set = TensorDataset(train_g, train_label, train_f, train_idx)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_g = torch.FloatTensor(np.array(g_list)[test_idx])
    test_label = torch.FloatTensor(np.array(label)[test_idx])
    test_f = torch.FloatTensor(np.array(f_list)[test_idx])
    test_idx = torch.FloatTensor(np.array(test_idx))
    test_set = TensorDataset(test_g, test_label, test_f, test_idx)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator


def accuracy(output, labels):
    #pdb.set_trace()
    y_pred = torch.round(output)
    # preds = output.argmax(1, keepdim=True)
    correct = y_pred.eq(labels.view_as(y_pred)).sum()
    acc = correct.float() / labels.shape[0]
    return acc


if __name__ == '__main__':
    train_idx_fold = np.load(config.train_idx_fold, allow_pickle=True)
    test_idx_fold = np.load(config.test_idx_fold, allow_pickle=True)
    print(train_idx_fold[0], len(train_idx_fold[0]))
    print(test_idx_fold[0], len(test_idx_fold[0]))

