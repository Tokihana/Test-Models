import torch
import torch.nn as nn
# third-party dependencies
from timm.layers import DropPath, Mlp
# local dependencies
from .ir50_stage3 import iresnet50_stage3

class RepeatAttention(nn.Module):
    def __init__(self, dim: int, 
                num_heads: int=8,
                qkv_bias: bool=False,
                qk_norm: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                norm_layer: nn.Module=nn.LayerNorm,):
        super(RepeatAttention, self).__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5
        
        self.wq, self.wk, self.wv = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x (BNC) -> B1C -> BNC'''
        B, N, C = x.shape
        # B1C -> B1H(H/C) -> BH1(H/C)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(H/C) -> BHN(H/C)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # qk norm
        q, k = self.q_norm(q), self.k_norm(k)
        
        # attn: BH1(C/H) @ BH(C/H)N -> BH1N -> BHNN
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn.repeat(1, 1, N, 1))
        
        # BHNN @ BHN(C/H) -> BHN(C/H) -> BNC
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
               
class RepeatAttentionBlock(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                has_mlp: bool=True):
        super(RepeatAttentionBlock, self).__init__()
        self.has_mlp = has_mlp
        self.norm1 = norm_layer(dim)
        self.attn = RepeatAttention(dim, 
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, 
                           hidden_features=int(dim*mlp_ratio), 
                           act_layer=act_layer, 
                           drop=proj_drop)
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class ExpandCLSBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int=8,
                 mlp_ratio: float=4.,
                 qkv_bias: bool=False,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 drop_path: float=0.,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module=nn.LayerNorm,
                 has_mlp: bool=True):
        super(ExpandCLSBlock, self).__init__()
        self.has_mlp = has_mlp
        self.norm1 = norm_layer(dim)
        self.attn = CLSAttention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim,
                           hidden_features=int(dim*mlp_ratio),   
                           act_layer=act_layer, 
                           drop=proj_drop)
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn. Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x (BNC)'''
        B, N, C = x.shape
        x = x + self.drop_path1(self.attn(self.norm1(x)).expand(-1, N, -1))
        if self.has_mlp:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class RepeatCLSBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int=8,
                 mlp_ratio: float=4.,
                 qkv_bias: bool=False,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 drop_path: float=0.,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module=nn.LayerNorm,
                 has_mlp: bool=True):
        super(RepeatCLSBlock, self).__init__()
        self.has_mlp = has_mlp
        self.norm1 = norm_layer(dim)
        self.attn = CLSAttention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim,
                           hidden_features=int(dim*mlp_ratio),   
                           act_layer=act_layer, 
                           drop=proj_drop)
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn. Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x (BNC)'''
        B, N, C = x.shape
        x = x + self.drop_path1(self.attn(self.norm1(x)).repeat(1, N, 1))
        if self.has_mlp:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    
def get_RepeatCLSFER(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                                      num_classes=config.MODEL.NUM_CLASS, 
                                      depth=config.MODEL.DEPTH, 
                                      mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,
                                      proj_drop=config.MODEL.PROJ_DROP,
                                      drop_path=config.MODEL.DROP_PATH,
                                     block=RepeatCLSBlock)
    return model

def get_ExpandCLSFER(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                                      num_classes=config.MODEL.NUM_CLASS, 
                                      depth=config.MODEL.DEPTH, 
                                      mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,
                                      proj_drop=config.MODEL.PROJ_DROP,
                                      drop_path=config.MODEL.DROP_PATH,
                                     block=ExpandCLSBlock)
    return model

def get_RepeatAttentionCLSFER(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                                      num_classes=config.MODEL.NUM_CLASS, 
                                      depth=config.MODEL.DEPTH, 
                                      mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,
                                      proj_drop=config.MODEL.PROJ_DROP,
                                      drop_path=config.MODEL.DROP_PATH,
                                     block=RepeatAttentionBlock)
    return model