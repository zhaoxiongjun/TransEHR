import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import os
import sys
sys.path.append('../')

import transEHR
from transEHR.utils import random_seed
from transEHR.modeling_transtab import TransEHRClassifier
from transEHR.train_multigpu import Trainer

from torch.utils.data.distributed import DistributedSampler


# set random seed
random_seed(42)


# 1) Init
torch.distributed.init_process_group(backend="nccl")

# 2） Configure the process for each gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# print(device)

# load dataset by passing data name and task name
train_dataset, val_dataset, test_dataset, num_train_set, cat_cols, num_cols, bin_cols = transEHR.load_data('../data/mimic-iii/', 'in-hospital-mortality')


# build transEHR classifier model
model = TransEHRClassifier(
        categorical_columns = cat_cols,
        numerical_columns = num_cols,
        binary_columns = bin_cols,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device=device,
        )

# 3） Assemble model to DDP
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

# specify training arguments, take validation loss for early stopping
training_arguments = {
    'num_epoch':100,
    'batch_size':16,   #6,
    'lr':1e-4,
    'eval_metric':'auc',
    'eval_less_is_better':False,
    'output_dir':'../checkpoint',
    'num_workers': 3,
    'warmup_ratio':None,
    'warmup_steps':None,
    'num_train_set':num_train_set,
}



trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        **training_arguments,
    )

trainer.train()
