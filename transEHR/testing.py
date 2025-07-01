import os
import time

import torch
import numpy as np
import pandas as pd
from tqdm.autonotebook import trange
from tqdm import tqdm
from loguru import logger

from . import constants
from .modeling_transtab import TransTabFeatureExtractor
from .utils import SupervisedTrainCollator
from .utils import DataLoaderX
from .metrics import print_metrics_binary

class Testing:
    def __init__(self,
        model,
        test_set_list,
        ckpt_dir='./ckpt',
        collate_fn=None,
        batch_size=32,
        num_workers=8,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        '''args:
        test_set_list: a list of test PatientDataset
        ckpt_dir: model save dir
        num_workers: how many workers used to process dataloader.
       
        '''
        self.model = model

        if not isinstance(test_set_list, list): test_set_list = [test_set_list]
        self.test_set_list = test_set_list
        
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )
        self.testloader_list = [
            self._build_dataloader(testset, batch_size, collator=self.collate_fn, num_workers=num_workers) for testset in test_set_list
        ]
       
        self.ckpt_dir = ckpt_dir
      

    def predict(self):
        start_time = time.time()
        if os.path.exists(self.ckpt_dir):
            # load checkpoints
            # logger.info(f'load best at last from {self.ckpt_dir}')
            # state_dict = torch.load(os.path.join(self.ckpt_dir, constants.WEIGHTS_NAME), map_location='cpu')
            # self.model.load_state_dict(state_dict)

            self.model.load(self.ckpt_dir)
        else:
            print('Error: ckpt_dir not exits !')
            return
        
        self.model.eval()
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in tqdm(self.testloader_list[dataindex]):
                if data[2] is not None:
                    label = data[2]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1], data[2])
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

          
            y_test = np.concatenate(y_test, 0)
            print_metrics_binary(y_true=y_test, predictions=pred_all)
        
        print("END Predict, cost time: {:.1f} secs".format(time.time()-start_time))


    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=False):
        trainloader = DataLoaderX(
            trainset,
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader
