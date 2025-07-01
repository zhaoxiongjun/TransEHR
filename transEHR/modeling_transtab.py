import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transformers import BertConfig, BertModel

from . import constants

class TransTabWordEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names, categorical and binary features.
    '''
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)
        return embeddings

class TransTabNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        '''args:
        num_col_emb: numerical column embedding, (# numerical columns, emb_dim)
        x_num_ts: numerical features, (bs, emb_dim)
        num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        '''
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class TransTabFeatureExtractor:
    r'''
    Process input dataframe to input indices towards transtab encoder,
    usually used to build dataloader for paralleling loading.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        '''args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        disable_tokenizer_parallel: true if use extractor for collator function in torch.DataLoader
        ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
            if set `true`, the duplicate cols will be deleted, else throws errors.
        '''
        if os.path.exists('./transEHR/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./transEHR/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./transEHR/tokenizer')

        # Using Bert Pre-train Model
        # self.tokenizer = BertTokenizerFast.from_pretrained('/home/zhaoxj/PhysioNet/Time-TransEHR/Bio_ClinicalBERT')
       
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # check if column exists overlap
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, shuffle=False) -> Dict:
        '''
        Parameters
        ----------
        x: pd.DataFrame 
            with column names and features.

        shuffle: bool
            if shuffle column order during the training.

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
                'x_cat_input_ids': tensor contains categorical column + feature ids,
                'x_bin_input_ids': tesnor contains binary column + feature ids,
            }
        '''
        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
            'x_bin_input_ids':None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c.lower() in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c.lower() in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c.lower() in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        # TODO:
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill Nan with zero or -1#这里保留了数值特征，用0用于后面乘
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols]
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x[cat_cols].astype(str)
            x_cat = x_cat.apply(lambda x: x.name + ' '+ x) * x_mask # mask out nan features
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_cat_ts['input_ids'].shape[1] > 0: # not all false
                encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
                encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols] # x_bin should already be integral (binary values in 0 & 1)
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0: # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        '''save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
        '''update cat/num/bin column maps.
        '''
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

class TransTabFeatureProcessor(nn.Module):
    r'''
    Process inputs from feature extractor to map them to embeddings.
    '''
    def __init__(self,
        vocab_size=None,
        hidden_dim=768,  #128
        hidden_dropout_prob=0,
        pad_token_id=0,
        device='cuda:0',
        ) -> None:
        '''args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        '''
        super().__init__()
        self.word_embedding = TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
            )
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1,keepdim=True).to(embs.device)
            return embs

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None,
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs,
        ) -> Tensor:
        '''args:
        x: pd.DataFrame with column names and features.
        shuffle: if shuffle column order during the training.
        num_mask: indicate the NaN place of numerical features, 0: NaN 1: normal.
        '''
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        if x_num is not None and num_col_input_ids is not None:
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device)) # number of cat col, num of tokens, embdding size
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            num_feat_embedding = self.align_layer(num_feat_embedding)

        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0: # all false, pad zero
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0],dtype=int)[:,None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # concat all embeddings
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            # att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
            att_mask_list += [torch.where(x_num.to('cpu') == 0, torch.tensor(0), torch.tensor(1))]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]  
            att_mask_list += [cat_att_mask.to('cpu')] # for multi-gpu
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding] 
            att_mask_list += [bin_att_mask.to('cpu')] # for multi-gpu
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}
      

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TransTabTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask= None, src_key_padding_mask= None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x

class TransTabInputEncoder(nn.Module):
    '''
    Build a feature encoder that maps inputs tabular samples to embeddings.
    
    Parameters:
    -----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds 
        multiple workers and the tokenizer builds multiple workers at the same time.

    hidden_dim: int
        the dimension of hidden embeddings.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    '''
    def __init__(self,
        feature_extractor,
        feature_processor,
        device='cuda:0',
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        Encode input tabular samples into embeddings.

        Parameters
        ----------
        x: pd.DataFrame
            with column names and features.        
        '''
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

class TransTabEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
            TransTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )
        if num_layer > 1:
            encoder_layer = TransTabTransformerLayer(d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
                )
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        '''args:
        embedding: bs, num_token, hidden_dim
        '''
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

class TransTabLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits

class TransEHRLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        clf_dropout=0.2,
        hidden_dim=768) -> None:
        super().__init__()
        if num_class <= 2:
            # self.fc = nn.Linear(128, 1)
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(128, num_class)
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.dropout = nn.Dropout(clf_dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = self.norm(x)
        # x = F.relu(self.dropout(self.linear1(x)))
        logits = self.fc(x)
        return logits


class TransTabCLSToken(nn.Module):
    '''add a learnable cls token embedding at the end of each sequence.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        # outputs = {'hidden_states': embedding} for bert pre-train model
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        # for bert pre-train model
        # outputs = {'hidden_states': embedding} 
        # outputs['attention_mask'] = attention_mask.unsqueeze(1).unsqueeze(1)

        outputs['attention_mask'] = attention_mask
        return outputs

class TransTabModel(nn.Module):
    '''The base transtab model for downstream tasks like contrastive learning, binary classification, etc.
    All models subclass this basemodel and usually rewrite the ``forward`` function. Refer to the source code of
    :class:`transtab.modeling_transtab.TransTabClassifier` or :class:`transtab.modeling_transtab.TransTabForCL` for the implementation details.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabModel model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=768,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None:
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )
        
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
            )

        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            )

        ### Use Bert Pre-train Model  ###
        # self.encoder = AutoModel.from_pretrained("../Bio_ClinicalBERT")
        # set BERT config
        # bert_config = BertConfig.from_pretrained('/home/zhaoxj/PhysioNet/Time-TransEHR/Bio_ClinicalBERT')
        # bert_config.num_hidden_layers = num_layer
        # # config.output_attentions = True 
        # # config.output_hidden_states = True

        # # load Bio_ClinicalBERT of pre num_layers
        # self.encoder = BertModel.from_pretrained('/home/zhaoxj/PhysioNet/Time-TransEHR/Bio_ClinicalBERT', config=bert_config).encoder

        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        '''Extract the embeddings based on input tables.

        Parameters
        ----------
        x: pd.DataFrame
            a batch of samples stored in pd.DataFrame.

        y: pd.Series
            the corresponding labels for each sample in ``x``. ignored for the basemodel.

        Returns
        -------
        final_cls_embedding: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        '''
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        # go through transformers, get final cls embedding
        encoder_output = self.encoder(**embeded)

        # get cls token
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding
     

    def load(self, ckpt_dir):
        '''Load the model state_dict and feature_extractor configuration
        from the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to load.

        Returns
        -------
        None

        '''
        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        '''Save the model state_dict and feature_extractor configuration
        to the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to save.

        Returns
        -------
        None

        '''
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        '''Update the configuration of feature extractor's column map for cat, num, and bin cols.
        Or update the number of classes for the output classifier layer.

        Parameters
        ----------
        config: dict
            a dict of configurations: keys cat:list, num:list, bin:list are to specify the new column names;
            key num_class:int is to specify the number of classes for finetuning on a new dataset.

        Returns
        -------
        None

        '''

        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

def dot_attention(q, z, mask):
    # q: query vector (shape: [batch_size, query_dim])
    # z: state trajectories (shape: [batch_size, max_time_steps, state_dim])
    # mask: mask tensor (shape: [batch_size, max_time_steps])
    
    # Calculate attention scores
    scores = torch.bmm(z, q.unsqueeze(-1)).squeeze(-1)  # shape: [batch_size, max_time_steps]
    scale = q.shape[-1] ** -0.5
    scores = scores/scale
    # Apply mask to attention scores
    masked_scores = scores.masked_fill(mask == 0, -np.inf)  # mask out padding time steps
    alpha = torch.softmax(masked_scores, dim=-1)  # shape: [batch_size, max_time_steps]
    
    # Calculate context vector
    context = torch.bmm(alpha.unsqueeze(1), z).squeeze(1)  # shape: [batch_size, state_dim]
    
    return context, alpha

class SimpleAttn(nn.Module):
    def __init__(self, hidden_dim=768, scale=10, attn_type='dot'):
        super(SimpleAttn, self).__init__()
        
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Parameter(torch.randn(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)

        self.scale = hidden_dim ** -0.5
        self.attn_type = attn_type

    # batch_size * sent_l * dim
    def forward(self, seq_embs, mask):
        '''
        Args:
            seq_embs: word embedding, batch_size, max_len, hidden_dim
            mask: mask of seqs, batch_size, max_len
        attention:
            score = v h
            att = softmax(score)
        '''
        hidden_vecs = seq_embs
        if self.attn_type == 'dot':
            inter_out = hidden_vecs
        else:
            inter_out = torch.tanh(self.attn_linear(hidden_vecs))
        #batch * max_len
        scores = torch.matmul(inter_out, self.V).squeeze(-1)
        scores = scores/self.scale
        #Mask the padding values
       
        scores = scores.masked_fill(mask == 0, -np.inf)
        #Softmax, batch_size*1*max_len
        attn = self.softmax(scores).unsqueeze(1)
        #weighted sum, batch_size*hidden_dim
        context_vec = torch.bmm(attn, hidden_vecs).squeeze(1)
        context_vec = self.dropout(context_vec)

        return context_vec, attn

class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """
    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)

class  TimeUpdataBlock(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8, batch_first=True, device=None, dtype=None,):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TimeUpdataBlock, self).__init__()

        self.input_self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=batch_first,
                                            **factory_kwargs)
        
        self.input_proj = nn.Linear(hidden_dim*2, hidden_dim, bias=False, **factory_kwargs)
        self.input_ff = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.state_out_to_gate = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.learned_ema_beta = nn.Parameter(torch.randn(hidden_dim))

        self.proj_gate = RecurrentStateGate(hidden_dim)
        self.ff_gate = RecurrentStateGate(hidden_dim)
        self.state_ff = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
    
    def forward(self, x, state=None, mask=None):
        batch, hidden_dim, device = x.shape[0], x.shape[-1], x.device
        if state == None:
            state = torch.zeros((batch, hidden_dim), device=device)

        key_padding_mask = ~mask.bool()
        input_attn = self.input_self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        state_attn, _ = dot_attention(state, x, mask)

        projected_input = self.input_proj(torch.concat((input_attn[:,0,:], state_attn), dim=-1))

        input_residual = projected_input + x[:,0,:]
        output = self.input_ff(input_residual) + input_residual

        ## LSTM gate
        state_residual = self.proj_gate(projected_input, state)
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)
        ## Fixed gate
        # z = self.state_out_to_gate(projected_input)
        # learned_ema_decay = self.learned_ema_beta.sigmoid()
        # states = learned_ema_decay * state + (1 - learned_ema_decay) * z

        return output, next_state

class TransEHRClassifier(TransTabModel):
    '''The classifier model subclass from :class:`transtab.modeling_transtab.TransTabModel`.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabClassifier model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=768,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:1',
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        # self.attn = SimpleAttn(hidden_dim=hidden_dim, attn_type='dot')
        self.TUB = TimeUpdataBlock(hidden_dim=hidden_dim, nhead=num_attention_head, device=device)
        self.tub_to_clf = nn.Linear(hidden_dim*2, hidden_dim)
        self.clf = TransEHRLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.to(device)

    def forward(self, X, X_len, y=None):
        '''Make forward pass given the input feature ``x`` and label ``y`` (optional).

        Parameters
        ----------
        X: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of TransTabFeatureExtractor.
        X_len: list
            the actual seq len of diff sample, such as the number of row time records of a patient.  
        y: pd.Series
            the corresponding labels for each sample in ``x``. if label is given, the model will return
            the classification loss by ``self.loss_fn``.

        Returns
        -------
        logits: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        loss: torch.Tensor or None
            the classification loss.

        '''
        # if isinstance(X, dict):
        #     # input is the pre-tokenized encoded inputs
        #     all_inputs = X
        # else:
        #     raise ValueError(f'TransTabClassifier takes inputs with dict, find {type(X)}.')
        all_inputs = X
        # 需要改，加for循环得到每一个的encoder——output，加attention后再进行分类。或LSTM
        # for each time step, do embedding and encoding
        static_output, time_outputs = [], []
        attn_inputs = []
        state_i = None
        for i in range(len(all_inputs)):
            inputs = all_inputs[i]
            outputs = self.input_encoder.feature_processor(**inputs)
            outputs = self.cls_token(**outputs)

            # go through transformers, get the first cls embedding in each time step
            encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim
            # encoder_output = self.encoder(**outputs)[0] for bert pre-train model
            
            # go through timeupdate block
            if i == 0:  # static feature encoder
                static_output = encoder_output[:,0,:]
            else:
                encoder_output_i, state_i = self.TUB(encoder_output, state_i, mask=outputs['attention_mask'])
                time_outputs.append(encoder_output_i)
            # attn_inputs.append(encoder_output[:,0,:])

        time_outputs = torch.stack(time_outputs, dim=1) # bs, max_time_step+1, hidden_dim
        # attn_inputs = torch.stack(attn_inputs, dim=1)

        # attention mask
        clf_input = []
        # bs_mask = torch.zeros((len(X_len), max(X_len)+1)).to(self.device) # bs, max_time_step
        for i, idx in enumerate(X_len):
            # bs_mask[i, :idx+1] = 1
            clf_input.append(time_outputs[i,idx-1,:])

        # attn_output, attn_scores = self.attn(attn_inputs, bs_mask) # bs, hidden_dim
        # classifier
        clf_input = torch.stack(clf_input, dim=0) # bs, hidden_dim

        to_clf = self.tub_to_clf(torch.concat((static_output, clf_input), dim=-1))
        logits = self.clf(to_clf)

        if y is not None:
            # compute classification loss
            if self.num_class == 2:
                y_ts = torch.tensor(y.values).to(self.device).float()
                loss = self.loss_fn(logits.flatten(), y_ts)
                # loss = py_sigmoid_focal_loss(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss