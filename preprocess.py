from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import KFold, StratifiedKFold
import pdb

"""
def get_target_subject(file_path):
    ###get_control_subject###
    data = pd.read_csv(file_path, delimiter=',')
    control_data = data[data['cahalan']=='control']
    control_sub = control_data['subject'].values
    control_sub_visit = control_data['visit'].values
    visit_dic = {0: 'baseline',
                 1: 'followup_1y',
                 2: 'followup_2y',
                 3: 'followup_3y',
                 4: 'followup_4y',
                 5: 'followup_5y',
                 6: 'followup_6y'}
    control_sub_ls = [control_sub[i]+'_'+visit_dic[control_sub_visit[i]]+'.mat' for i in range(len(control_data))]
    ###find intersect of (control_subject, rsfmri_subject, dti_subject)###
    rsfmri_sub_ls = os.listdir('E:/graduate/Courses/PYSC399/rsfmri/')
    dti_sub_ls = os.listdir('E:/graduate/Courses/PYSC399/dti/')
    intersection_set = set.intersection(set(control_sub_ls), set(rsfmri_sub_ls), set(dti_sub_ls))
    target_sub_ls = sorted(list(intersection_set))
    # d = {'target_subject': target_sub_ls}
    # df = pd.DataFrame(data=d)
    # df.to_csv('E:/graduate/Courses/PYSC399/connectome/003/target_sub_list.csv', index=False)
    return target_sub_ls

def demographics_loader(demo_file, target_sub_list):
    demo = pd.read_csv(demo_file, delimiter=',')
    g_sex_label = [0] * len(target_sub_list)
    g_age_label = []
    subject_list = []
    for i, subject in enumerate(target_sub_list):
        subject_list.append(subject.split('_')[0]+'_'+subject.split('_')[1])
        sex = demo[(demo['subject'] == subject.split('_')[0]+'_'+subject.split('_')[1]) &
            (demo['visit'] == subject.split('_', 2)[2].split('.')[0]) & (demo['arm'] == 'standard')]['sex'].values
        age = demo[(demo['subject'] == subject.split('_')[0] + '_' + subject.split('_')[1])&
            (demo['visit'] == subject.split('_', 2)[2].split('.')[0]) & (demo['arm'] == 'standard')]['visit_age'].values
        g_sex_label[i] = 0 if sex == 'F' else 1
        g_age_label.append(age[0])
    d = {'subject_file': target_sub_list, 'subject': subject_list, 'sex': g_sex_label, 'age': g_age_label}
    df = pd.DataFrame(data=d)
    df.to_csv('E:/graduate/Courses/PYSC399/connectome/003/'+'control_subject_label.csv', index=False)
"""


def normalization(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = np.matmul(r_mat_inv.todense(), mx)
    mx = np.matmul(mx, r_mat_inv.todense())
    mx[np.isnan(mx)] = 0
    return mx

def get_fc_graph(fc_folder, sc_folder, label_file):
    target_subject = pd.read_csv(label_file)['subject_file'].values
    fc_list, sc_list = [], []
    for file in target_subject:
        fc = scipy.io.loadmat(fc_folder+file)['matrix']
        sc = scipy.io.loadmat(sc_folder+file)['matrix']
        # FC normalization
        fc = fc - np.eye(80)
        fc = np.log((1 + fc) / (1 - fc)) # fisher transform
        fc = normalize(fc+np.eye(80)) # normalization
        # SC normalization
        sc = np.log(sc+1)
        sc = normalize(sc)

        print('----------FC-------------',fc)
        print('----------SC-------------',sc)
        fc_list.append(fc)
        sc_list.append(sc)




    # print(target_subject)
    # for i in target_subject
    # print(fc_list)
    # print(target_subject)


if __name__ == '__main__':

    """ 
    graph_path = 'E:/graduate/Courses/PYSC399/connectome/001/adj/'
    label_file = 'E:/graduate/Courses/PYSC399/connectome/001/sex_label/sex_label.csv'
    subject_list_file = 'E:/graduate/Courses/PYSC399/connectome/001/subject/subject_list.csv'
    subject_full_list_file = 'E:/graduate/Courses/PYSC399/connectome/001/subject/subject_full_list.csv'
    train_idx_fold, test_idx_fold = data_split(subject_list_file, label_file)
    print(train_idx_fold)
    print('=========================================')
    print(test_idx_fold)
    
    demo_path = "E:/graduate/Courses/PYSC399/connectome/resources/demographics.csv"
    cahalan = "E:/graduate/Courses/PYSC399/connectome/resources/2019-10-02_cahalan.csv"
    label_path = 'E:/graduate/Courses/PYSC399/connectome/003/'+'control_subject_label.csv'
    fc_folder = 'E:/graduate/Courses/PYSC399/rsfmri/'
    sc_folder = 'E:/graduate/Courses/PYSC399/dti/'
    # target_sub_ls = get_target_subject(cahalan)
    # demographics_loader(demo_path, target_sub_ls)
    # data_split(label_path)
    """
    fc_folder = './raw_data/fc_adjacency/'
    sc_folder = './raw_data/sc_adjacency/'
    label_path = './003/control_subject_label.csv'
    get_fc_graph(fc_folder, sc_folder, label_path)

