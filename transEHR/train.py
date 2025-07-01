import os
import pdb
import math
import time
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
# from transformers.optimization import get_scheduler
from tqdm.autonotebook import trange
from loguru import logger

from . import constants
from .evaluator import get_eval_metric_fn, EarlyStopping
from .modeling_transtab import TransTabFeatureExtractor
from .utils import SupervisedTrainCollator
from .utils import get_parameter_names, get_scheduler
from .utils import DataLoaderX

class Trainer:
    def __init__(self,
        model,
        train_set_list,
        val_set_list=None,
        collate_fn=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-5,
        patience=10,
        eval_batch_size=32,
        num_train_set=0,  # number of training sample
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=False,
        ignore_duplicate_cols=False,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=7,
        is_fine_turn=False,
        **kwargs,
        ):
        '''args:
        train_set_list: a list of train PatientDataset
        val_set_list: a list of PatientDataset, same as train_set_list. if set None, do not do evaluation and early stopping
        patience: the max number of early stop patience
        num_workers: how many workers used to process dataloader. recommend to be 0 if training data smaller than 10000.
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        '''
        self.model = model
        if not isinstance(train_set_list, list): train_set_list = [train_set_list]
        if not isinstance(val_set_list, list): val_set_list = [val_set_list]

        self.train_set_list = train_set_list
        self.val_set_list = val_set_list
        
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )
        # sampler = WeightedRandomSampler(weights=train_set_list[0].get_weights(), num_samples=num_train_set, replacement=True)
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn, sampler=None, num_workers=num_workers, shuffle=True) for trainset in train_set_list
        ]
        if val_set_list is not None:
            self.valloader_list = [
                self._build_dataloader(valset, eval_batch_size, collator=self.collate_fn, sampler=None, num_workers=3, shuffle=False) for valset in val_set_list
            ]
        else:
            self.valloader_list = None
        # self.valloader_list = None
        
            
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(num_train_set, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
            'is_fine_turn': is_fine_turn,
            }
        # self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch*len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self):
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            num_train_steps = args['num_training_steps']
            logger.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)

        if args['is_fine_turn']:  ## for fine-turn
            print("####### Fine Turn #######")
            ckpt_dir = "../sepsis_checkpoint/epoch_6/"
            if os.path.exists(ckpt_dir):
                self.model.load(ckpt_dir)

        start_time = time.time()
        for epoch in trange(args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss_all = 0
            self.model.train()  # add 
            for dataindex in range(len(self.trainloader_list)):
                y_train, pred_list = [], []
                for data in self.trainloader_list[dataindex]:
                    label = data[2]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_train.append(label)

                    self.optimizer.zero_grad()
                    logits, loss = self.model(data[0], data[1], data[2])
                    loss.backward()
                    self.optimizer.step()

                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
                    
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    print('epoch: {}, ite: {}, train loss: {:.4f}'.format(epoch, ite, loss.item()))

                if len(pred_list)>0:
                    pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()
                y_train = np.concatenate(y_train, 0)
                train_eval_res = self.args['eval_metric'](y_train, pred_all)
            print('epoch: {}, train loss: {:.4f}, train {}: {:.6f} lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.args['eval_metric_name'], train_eval_res, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
            
            if self.val_set_list is not None:
                eval_res_list = self.evaluate(self.valloader_list)  # for testing
                # eval_res = np.mean(eval_res_list)
                eval_res, eval_loss = eval_res_list[0], eval_res_list[1]
                print('epoch: {}, test {}: {:.6f}, loss: {:.4f}'.format(epoch, self.args['eval_metric_name'], eval_res, eval_loss))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break
           
            if eval_res > 0.85:
                self.save_epoch_model(self.output_dir, epoch=epoch)
        # if os.path.exists(self.output_dir):
        #     if self.test_set_list is not None:
        #         # load checkpoints
        #         logger.info(f'load best at last from {self.output_dir}')
        #         state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
        #         self.model.load_state_dict(state_dict)
        #     self.save_model(self.output_dir)

        logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))

    def evaluate(self, loader_list):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        for dataindex in range(len(loader_list)):
            y_test, pred_list, loss_list = [], [], []
            loss_all = 0
            for data in loader_list[dataindex]:
                if data[2] is not None:
                    label = data[2]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1], data[2])
                if loss is not None:
                    # loss_list.append(loss.item())
                    loss_all+=loss.item()
                if logits is not None:
                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = np.concatenate(y_test, 0)
                eval_res = self.args['eval_metric'](y_test, pred_all)

            eval_res_list.append(eval_res)
            eval_res_list.append(loss_all)

        return eval_res_list

    def save_epoch_model(self, output_dir=None, epoch=0):
        output_dir = os.path.join(output_dir, 'epoch_{}'.format(epoch))

        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

        logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)

    
    def save_model(self, output_dir=None):
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)
        self.collate_fn.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))
        if self.args is not None:
            train_args = {}
            for k,v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(os.path.join(output_dir, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'polynomial',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, num_train_set, num_epoch, batch_size):
        total_step = 0
        # for trainset in train_set_list:
        #     x_train, _ = trainset
        #     total_step += np.ceil(len(x_train) / batch_size)
        total_step += np.ceil(num_train_set / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, sampler, num_workers=8, shuffle=False):
        trainloader = DataLoaderX(
            trainset,
            collate_fn=collator,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return trainloader
