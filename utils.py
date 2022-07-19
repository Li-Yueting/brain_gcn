import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
import config

import pdb
import logging
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generator(train_idx, test_idx, batch_size, task):

    # TODO:
    # load data
    label = pd.read_csv(config.label_file)[task.split('_')[0]].values
    g_list = np.load(config.JOINT_path, allow_pickle=True)
    m_list = np.load(config.Machine_path, allow_pickle=True)
    m_list = np.where(m_list == 'ge', 1.0, 0.0)[:, None]
    f_list = np.load(config.Concat_Feature_path, allow_pickle=True)

    # divide train
    train_g = torch.FloatTensor(np.array(g_list)[train_idx])
    train_f = torch.FloatTensor(np.array(f_list)[train_idx])
    train_m = torch.FloatTensor(np.array(m_list)[train_idx])
    train_label = torch.FloatTensor(np.array(label)[train_idx])
    train_idx = torch.FloatTensor(np.array(train_idx))
    # generate train data
    train_set = TensorDataset(train_g, train_label, train_f, train_m, train_idx)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # divide test
    test_g = torch.FloatTensor(np.array(g_list)[test_idx])
    test_f = torch.FloatTensor(np.array(f_list)[test_idx])
    test_m = torch.FloatTensor(np.array(m_list)[test_idx])
    test_label = torch.FloatTensor(np.array(label)[test_idx])
    test_idx = torch.FloatTensor(np.array(test_idx))
    # generate test data
    test_set = TensorDataset(test_g, test_label, test_f, test_m, test_idx)
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
    f_list = np.load(config.Concat_Feature_path, allow_pickle=True)
    print(f_list.shape)
    # train_idx_fold = np.load(config.train_idx_fold, allow_pickle=True)
    # test_idx_fold = np.load(config.test_idx_fold, allow_pickle=True)
    # print(train_idx_fold[0], len(train_idx_fold[0]))
    # print(test_idx_fold[0], len(test_idx_fold[0]))

