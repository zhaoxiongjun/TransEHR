import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from collections import Counter

class PatientDataset(Dataset):
    def __init__(self, mode, root_dir, label_df):
        assert mode in ["train", "val", "test"]
        self.mode = mode

        # self.list_df = pd.read_csv(label_file)
        self.list_df = label_df
        if self.mode == "test":
            self.patient_path = [os.path.join(root_dir,'test',file_name) for file_name in self.list_df['stay']]
        else:
            self.patient_path = [os.path.join(root_dir,'train',file_name) for file_name in self.list_df['stay']]
        
        self.class_counts = Counter(label_df.iloc[:,-1])

    def __len__(self):
        return len(self.list_df)
    
    def __getitem__(self, index):
        p_path = self.patient_path[index]
        
        time_feats = pd.read_csv(p_path)
        label = self.list_df.iloc[index:index+1, -1]
        # label = self.list_df['y_true'][index]
        static_feats = self.list_df.iloc[index:index+1, 1:-1]
        # static_feats = self.list_df.iloc[index, 1:-1]
        return static_feats, time_feats, label  #torch.tensor(label, dtype=torch.long)
    
    def get_weights(self):
        n_samples = len(self.list_df)
        weights = [0] * n_samples
        class_weights = {0: 1.0 / self.class_counts[0], 1: 1.0 / self.class_counts[1]}
        for i in range(n_samples):
            class_weight = class_weights[self.list_df.iloc[i, -1]]
            weights[i] = class_weight
        return weights

def load_data(dataname, taskname, dataset_config=None, seed=123):
    '''Load datasets from the local device.

    Parameters
    ----------
    dataname: str
        the dataset name intended to be loaded from the directory to the local dataset.
    
    taskname: str 
        the task to do
    
    dataset_config: dict
        the dataset configuration to specify for loading. Please note that this variable will
        override the configuration loaded from the local files or from the openml.dataset.
    
    seed: int
        the random seed set to ensure the fixed train/val/test split.

    Returns
    -------
    train_list: list or tuple
        the train dataset list, contains static features and label.

    val_list: list or tuple
        the validation dataset list, contains static features and label.

    test_list: list
        the test dataset list, contains static features and label.

    cat_col_list: list
        the list of categorical column names.

    num_col_list: list
        the list of numerical column names.

    bin_col_list: list
        the list of binary column names.

    '''
    if isinstance(dataname, str):
        # load a single dataset
        return load_single_data(dataname=dataname, taskname=taskname, dataset_config=dataset_config)
    
    if isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        train_list, val_list, test_list = [], [], []
        trainlens = 0
        for dataname_, taskname_ in zip(dataname, taskname):
            trainset, valset, testset, trainlen, cat_cols, num_cols, bin_cols = \
                load_single_data(dataname_, taskname=taskname_, dataset_config=dataset_config)

            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)

            train_list.append(trainset)
            val_list.append(valset)
            # test_list.append(testset)
            test_list = testset
            trainlens += trainlen
        
        num_col_list = list(set(num_col_list))
        cat_col_list = list(set(cat_col_list))
        bin_col_list = list(set(bin_col_list))
        
        return train_list, val_list, test_list, trainlens, cat_col_list, num_col_list, bin_col_list

def load_single_data(dataname, taskname, dataset_config=None):

    print('####'*10)
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname} for {taskname} task')

        task_path = os.path.join(dataname, taskname)

        # load column info
        catfile = os.path.join(task_path, 'categorical_feature.txt')
        if os.path.exists(catfile):
            with open(catfile,'r') as f: cat_cols = [x.strip().lower() for x in f.readlines()]
        else:
            cat_cols = []
        
        numfile = os.path.join(task_path, 'numerical_feature.txt')
        if os.path.exists(numfile):
            with open(numfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []

        bnfile = os.path.join(task_path, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        
        # build train/val/test Dataset
        trainfile = os.path.join(task_path, 'train_listfile.csv')
        if os.path.exists(trainfile):
            train_df = pd.read_csv(trainfile)
            train_dataset = PatientDataset(root_dir=task_path, mode="train", label_df=train_df)

        valfile = os.path.join(task_path, 'val_listfile.csv')
        if os.path.exists(valfile):
            val_df = pd.read_csv(valfile)
            val_dataset = PatientDataset(root_dir=task_path, mode="val", label_df=val_df)

        testfile = os.path.join(task_path, 'test_listfile.csv')
        if os.path.exists(testfile):
            test_df = pd.read_csv(testfile)
            test_dataset = PatientDataset(root_dir=task_path, mode="test", label_df=test_df)

    else: 
        print(f'{dataname} not exits')


    # update cols by loading dataset_config
    # if dataname in dataset_config:
    #     data_config = dataset_config[dataname]

    #     if 'bin' in data_config:
    #         bin_cols = data_config[taskname]['bin']
            
    #     if 'cat' in data_config:
    #         cat_cols = data_config[[dataname]]['cat']

    #     if 'num' in data_config:
    #         num_cols = data_config[dataname]['num']


    data_nums = len(train_df)+len(val_df)+len(test_df)
    feats = len(cat_cols) + len(bin_cols) + len(num_cols)

    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(data_nums, feats, len(cat_cols), len(bin_cols), len(num_cols)))
    return (train_dataset), (val_dataset), (test_dataset), len(train_df), cat_cols, num_cols, bin_cols

