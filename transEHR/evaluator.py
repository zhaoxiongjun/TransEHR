from collections import defaultdict
import os
import pdb

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from . import constants

def evaluate(ypred, y_test, metric='auc', seed=123, bootstrap=False):
    np.random.seed(seed)
    eval_fn = get_eval_metric_fn(metric)
    res_list = []
    stats_dict = defaultdict(list)
    if bootstrap:
        for i in range(10):
            sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
            sub_ypred = ypred[sub_idx]
            sub_ytest = y_test.iloc[sub_idx]
            try:
                sub_res = eval_fn(sub_ytest, sub_ypred)
            except ValueError:
                print('evaluation went wrong!')
            stats_dict[metric].append(sub_res)
        for key in stats_dict.keys():
            stats = stats_dict[key]
            alpha = 0.95
            p = ((1-alpha)/2) * 100
            lower = max(0, np.percentile(stats, p))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print('{} {:.2f} mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
            if key == metric: res_list.append((upper+lower)/2)
    else:
        res = eval_fn(y_test, ypred)
        res_list.append(res)
    return res_list

def get_eval_metric_fn(eval_metric):
    fn_dict = {
        'acc': acc_fn,
        'auc': auc_fn,
        'mse': mse_fn,
        'val_loss': None,
    }
    return fn_dict[eval_metric]

def acc_fn(y, p):
    y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)

def auc_fn(y, p):
    return roc_auc_score(y, p)

def mse_fn(y, p):
    return mean_squared_error(y, p)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print     
            less_is_better (bool): If True (e.g., val loss), the metric is less the better.       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0: # no early stop
            self.early_stop = False
            return
        
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, constants.WEIGHTS_NAME))
        self.val_loss_min = val_loss

