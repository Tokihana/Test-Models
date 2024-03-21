import torch
import torch.nn as nn
# third-party dependencies
from timm.layers import DropPath


class CLSAttention(nn.Module):
    def __init__(self, dim: int, 
                num_heads: int=8,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.):
        super(CLSAttention, self).__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5
        
        self.wq, self.wk, self.wv = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x (B, N, C) -> return (B, 1, C)'''
        B, N, C = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k =  self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # attn: BH1(C/H) @ BH(C/H)N -> BH1N
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = attn.softmax(dim=-1) # dim that will compute softmax, every slice along this dim will sum to 1, for attn case, is N
        attn = self.attn_drop(attn)
        
        # BH1N @ BHN(C/H) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
               
class CLSBlock(nn.Module):
    '''plus attn with identity shortcut
    x (B, N, C) -> return (B, 1, C)'''
    def __init__(self, dim: int,
                num_heads: int=8,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm):
        super(CLSBlock, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = CLSAttention(dim, 
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1, ...] + self.drop_path(self.attn(self.norm(x)))
    
class NonMultiCLSBlock(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm):
        super(CLSBlock, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = CLSAttention(dim, 
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[:, 0:1, ...] = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm(x)))
        return x
    
    
class CLSViT(nn.Module):
    '''ViT with CLSBlock and no pyramid structure'''
    def __init__(self):
        super(CLSViT, self).__init__()
        self.block
    def forward(self, x):
        return