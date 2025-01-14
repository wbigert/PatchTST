__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
          
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # Add variables to self
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.individual = individual
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.norm = norm
        self.attn_dropout = attn_dropout
        self.act = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res, res_embeddings = self.model_res(res_init)
            trend, trend_embeddings = self.model_trend(trend_init)
            x = res + trend
            embeddings = res_embeddings + trend_embeddings  # You might need to adjust this operation based on how you want to combine embeddings
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x, embeddings, high_dim_embeddings = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x, embeddings, high_dim_embeddings
    
    def print_model_parameters(self):
            print("Model Parameters:")
            print(f"c_in: {self.c_in}")
            print(f"context_window: {self.context_window}")
            print(f"target_window: {self.target_window}")
            print(f"n_layers: {self.n_layers}")
            print(f"n_heads: {self.n_heads}")
            print(f"d_model: {self.d_model}")
            print(f"d_ff: {self.d_ff}")
            print(f"dropout: {self.dropout}")
            print(f"fc_dropout: {self.fc_dropout}")
            print(f"head_dropout: {self.head_dropout}")
            print(f"individual: {self.individual}")
            print(f"patch_len: {self.patch_len}")
            print(f"stride: {self.stride}")
            print(f"padding_patch: {self.padding_patch}")
            print(f"revin: {self.revin}")
            print(f"affine: {self.affine}")
            print(f"subtract_last: {self.subtract_last}")
            print(f"decomposition: {self.decomposition}")
            print(f"kernel_size: {self.kernel_size}")
            print(f"max_seq_len: {self.max_seq_len}")
            print(f"d_k: {self.d_k}")
            print(f"d_v: {self.d_v}")
            print(f"norm: {self.norm}")
            print(f"attn_dropout: {self.attn_dropout}")
            print(f"act: {self.act}")
            print(f"key_padding_mask: {self.key_padding_mask}")
            print(f"padding_var: {self.padding_var}")
            print(f"attn_mask: {self.attn_mask}")
            print(f"res_attention: {self.res_attention}")
            print(f"pre_norm: {self.pre_norm}")
            print(f"store_attn: {self.store_attn}")
            print(f"pe: {self.pe}")
            print(f"learn_pe: {self.learn_pe}")
            print(f"pretrain_head: {self.pretrain_head}")
            print(f"head_type: {self.head_type}")
            

            print("\nModel Weights (First 5 values for brevity):")
            for name, weight in self.state_dict().items():
                # Adjust to safely print first few values of the tensor
                values_to_print = weight.flatten()[:5]  # Flatten and take first 5 values
                print(f"Layer: {name} | Size: {weight.size()} | Values: {values_to_print}...")