from typing import Optional
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale

class StarAttention(nn.Module):
    def __init__(self, dim: int, 
                num_heads: int=8,
                qkv_bias: bool=False,
                qk_norm: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                norm_layer: nn.Module=nn.LayerNorm,):
        super().__init__()
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        head_dim: int = dim // num_heads
        self.scale: float = dim // head_dim ** -0.5
        self.tanh = nn.Tanh()
        
        self.Wq, self.Wk, self.Wv = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias)
        if qk_norm:
            self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        # BNC -> BNH(C/H) -> BHN(C/H)
        q = self.Wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.Wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.Wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        
        attn = self.tanh(q * k).sum(axis=-1, keepdim=True) * self.scale
        attn = self.attn_drop(attn)
        
        x = (attn * v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
        
        
class StarBlock(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
                qk_norm: bool=False,
                init_values: Optional[float] = None,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                has_mlp: bool=True,
                add_to_patch: bool=False):
        super(StarBlock, self).__init__()
        self.has_mlp = has_mlp
        self.add_to_patch = add_to_patch
        self.norm1 = norm_layer(dim)
        self.attn = StarAttention(dim, 
                                 num_heads=num_heads,
                                 qk_norm=qk_norm,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, 
                           hidden_features=int(dim*mlp_ratio), 
                           act_layer=act_layer, 
                           drop=proj_drop)
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.has_mlp:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x