import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import os
import sys
sys.path.append('../')

import transEHR
from transEHR.utils import random_seed
from transEHR.modeling_transtab import TransEHRClassifier
from transEHR.train import Trainer
# set random seed
random_seed(42)


# load dataset by passing data name and task name
train_dataset, val_dataset, test_dataset, num_train_set, cat_cols, num_cols, bin_cols = transEHR.load_data('../data/mimic-iii/', 'in-hospital-mortality')


# build transEHR classifier model


model = TransEHRClassifier(
        categorical_columns = cat_cols,
        numerical_columns = num_cols,
        binary_columns = bin_cols,
        num_class=2,
        hidden_dim=128,
        num_layer=3,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
    )


# logits, attn_scores, loss = model(data[0], data[1], data[2])
# print(logits, attn_scores, loss)


# specify training arguments, take validation loss for early stopping
training_arguments = {
    'num_epoch':100,
    'batch_size':32,
    'lr':1e-3,
    'eval_metric':'auc',
    'eval_less_is_better': False,
    'output_dir':'../checkpoint',
    'num_workers': 7,
    'warmup_ratio':None,
    'warmup_steps':600,
    'num_train_set':num_train_set,
}



trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        **training_arguments,
    )

trainer.train()
