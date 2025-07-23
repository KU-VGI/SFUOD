import math
import copy

import torch
import torch.nn.functional as F
from torch.nn.functional import relu, interpolate
from torch import nn


# from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


class Cross_attn_layer(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        #? Projection Layer는 별도로 DETR 내부에서 관리 (Naming 중요)
        #? self.embed_proj = nn.Linear(embed_dim, feat_dim)
        
        #* Attention Layers
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.weight_q = nn.Linear(feat_dim, feat_dim)
        self.weight_kv = nn.Linear(feat_dim, feat_dim*2)
        self.proj = nn.Linear(feat_dim, feat_dim)
        
        #* Normalization Layers
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        #* MLP Layers
        self.fc1 = nn.Linear(feat_dim, feat_dim*4)
        # self.act = nn.GELU()
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim*4, feat_dim)
        
        self._init_params()
    
    def _init_params(self):
        nn.init.xavier_uniform_(self.weight_q.weight, gain=1)
        nn.init.constant_(self.weight_q.bias, 0)
        
        nn.init.xavier_uniform_(self.weight_kv.weight, gain=1)
        nn.init.constant_(self.weight_kv.bias, 0)
        
        nn.init.xavier_uniform_(self.proj.weight, gain=1)
        nn.init.constant_(self.proj.bias, 0)
        
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, proj_x, obj_queries):
        #todo 1. Cross Attention
        B, N_hw, C = proj_x.shape   #* B: # of images, N: H*W, C=dimension (256)
        _, N, _ = obj_queries.shape
        # print('[Cross] Training Mode:', self.training)
        q = self.weight_q(self.norm1(obj_queries)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = self.weight_kv(self.norm1(proj_x)).reshape(B, N_hw, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # print('[Cross] QKV:', q.shape, k.shape, v.shape)    #* B,num_head,N,head_dim

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        # print('[Cross] attn_score:', attn.shape)    #* B,num_head,N,N
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = obj_queries + self.proj(x)
        
        # return x
        # print('[Cross] attn_out:', x.shape) #* B,N,C
        
        #todo 2. MLP 
        norm_x = self.norm2(x)
        x1 = self.act(self.fc1(norm_x))
        x2 = self.fc2(x1)
        
        out = x + x2
        
        # print('[Cross] mlp_out:', out.shape)    #* B,N,C
        
        return out

class Bi_Attention_layer(nn.Module):
    def __init__(self,feat_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.weight_q = nn.Linear(feat_dim, feat_dim)
        self.weight_kv = nn.Linear(feat_dim, feat_dim*2)
        self.proj = nn.Linear(feat_dim, feat_dim)
        
        #* Normalization Layers
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        self.norm3 = nn.LayerNorm(feat_dim)
        
        # * MLP Layers
        self.fc1 = nn.Linear(feat_dim, feat_dim*4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim*4, feat_dim)
        
    
    def forward(self, feat_cross, feat_detr):
        #todo: Input: DETR Embeddings (Source) & Cross-Attention Embeddings (Target)
        #todo 1. Dual Attention
        B, N, C = feat_detr.shape   #* B: # of images, N: Proposals, C=dimension (256)
        
        norm_source = self.norm1(feat_detr)
        norm_target = self.norm1(feat_cross)
        
        q = self.weight_q(norm_source).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # print('[Bi-Attn] Q:', q.shape)  #* B, num_head, N, head_dim
        combined = torch.cat([norm_source, norm_target], dim=1)
        # print('[Bi-Attn] combined:', combined.shape)    #* B, 2n, C
        
        _, comb_N, _ = combined.shape
        k, v = self.weight_kv(combined).reshape(B, comb_N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # print('[Bi-Attn] QKV:', q.shape, k.shape, v.shape)  #* B,num_head, N/2N, head_dim
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  #* B, num_head, N, 2N
        
        #* Ablation Study: Calculating Attention Score
        # inter_attn = attn.softmax(dim=-1)
        # x = inter_attn @ v
        
        #* Cross-Domain Attention Score
        intra_attn = attn.reshape(B, self.num_heads, N, 2, -1).softmax(dim=-2).reshape(B, self.num_heads, N, -1)
        x = intra_attn @ v
        
        
        #* Both: Global & Local
        # both_attn = inter_attn + intra_attn
        # x = both_attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = feat_cross + self.proj(x)
        # return x
        #todo 2. MLP 
        norm_x = self.norm2(x)
        x1 = self.act(self.fc1(norm_x))
        x2 = self.fc2(x1)
        
        out = x + x2
        
        return self.norm3(out)
        # return out