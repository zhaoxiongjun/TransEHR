import os
import random
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from .modeling_transtab import TransTabFeatureExtractor

TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}

class TrainCollator:
    '''A base class for all collate function used for TransTab training.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        self.feature_extractor=TransTabFeatureExtractor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def save(self, path):
        self.feature_extractor.save(path)
    
    def __call__(self, data):
        raise NotImplementedError

class SupervisedTrainCollator(TrainCollator):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        super().__init__(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def __call__(self, data):
        X_s = pd.concat([row[0] for row in data])
        X_t = [row[1] for row in data]
        X_t_lens = [x_t.shape[0] for x_t in X_t]
        # X_t_lens = [math.ceil(x * 0.9) for x in X_t_lens]
        y = pd.concat([row[2] for row in data])

        all_inputs = []
        input_0 = self.feature_extractor(X_s)
        all_inputs.append(input_0)

        for i in range(max(X_t_lens)):
            X_t_i = pd.DataFrame(columns=X_t[0].columns)
            for j in range(len(X_t)):
                if i < X_t_lens[j]:
                    X_t_i = X_t_i.append(X_t[j].iloc[i].to_dict(), ignore_index=True)
                else:
                    X_t_i = X_t_i.append({}, ignore_index=True)
                
            inputs = self.feature_extractor(X_t_i)

            all_inputs.append(inputs)


        return all_inputs, X_t_lens, y.astype('int64')


from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_scheduler(
    name,
    optimizer,
    num_warmup_steps = None,
    num_training_steps = None,
    ):
    '''
    Unified API to get any scheduler from its name.

    Parameters
    ----------
    name: str
        The name of the scheduler to use.

    optimizer: torch.optim.Optimizer
        The optimizer that will be used during training.

    num_warmup_steps: int
        The number of warmup steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    
    num_training_steps: int
        The number of training steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    '''
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == 'constant':
        return schedule_func(optimizer)
    
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")


    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


