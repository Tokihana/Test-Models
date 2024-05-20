# inline dependencies
from typing import Optional
# third-party dependencies
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock
from timm.models.vision_transformer import LayerScale
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
from functools import partial
# local dependencies
from .ir50 import iresnet50 # which return x2, x3, x4

### ---------------------------------------------
# utils
### ---------------------------------------------
class SE_block(nn.Module):
    def __init__(self, channels, rd_ratio=1./16, rd_divisor=8, bias=True):
        super().__init__()
        rd_chans = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv1d(channels, rd_chans, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(rd_chans)
        self.act_layer = nn.ReLU()
        self.fc2 = nn.Conv1d(rd_chans, channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        '''
        x (B, N, C)
        '''
        # squeeze, BNC -> B1C -> BC1
        x_se = x.mean(dim=1, keepdim=True).permute(0, 2, 1)
        x_se = self.sigmoid(self.fc2(self.act_layer(self.bn(self.fc1(x_se)))))
        x = x * x_se.permute(0, 2, 1)
        return x
    
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
            
class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        self.use_conv=use_conv
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        if self.use_conv:
            x = x.permute(0, -1, -2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        if self.use_conv:
            x = x.permute(0, -1, -2)
        return x


### ---------------------------------------------
# Class Attention
### ---------------------------------------------
class CLSAttention(nn.Module):
    '''
    Q_cls, K, V = W_Qx_cls, W_KX, W_VX
    attn = Q_cls @ K^T （B1C @ BCN -> B1N)
    out = attn @ V (B1N @ BNC -> B1C)
    '''
    def __init__(self, dim: int, 
                num_heads: int=8,
                qkv_bias: bool=False,
                qk_norm: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                norm_layer: nn.Module=nn.LayerNorm,):
        super(CLSAttention, self).__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5
        
        self.wq, self.wk, self.wv = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
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
        # qk norm
        q, k = self.q_norm(q), self.k_norm(k)
        
        # attn: BH1(C/H) @ BH(C/H)N -> BH1N
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = attn.softmax(dim=-1) # dim that will compute softmax, every slice along this dim will sum to 1, for attn case, is N
        attn = self.attn_drop(attn)
        
        # BH1N @ BHN(C/H) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    

### ---------------------------------------------
# AC-CAE Block
### ---------------------------------------------
class ClassAttentionBlock(nn.Module):
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
                mlp_layer: nn.Module = Mlp,
                has_mlp: bool=True,
                conv_mlp: bool=False,
                update: str='cat'):
        '''
        update: 'cat', 'additive', or 'star'
        '''
        super(NonMultiCLSBlock_catAfterMlp, self).__init__()
        self.has_mlp = has_mlp
        self.update = update
        self.norm1 = norm_layer(dim)
        self.attn = CLSAttention(dim, 
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_norm=qk_norm,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,
                                 norm_layer=norm_layer,)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = mlp_layer(in_features=dim, 
                                 hidden_features=int(dim*mlp_ratio), 
                                 act_layer=act_layer, 
                                 drop=proj_drop,
                                use_conv=conv_mlp)
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_cls = x[:, 0:1, ...] + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        if self.update == 'additive': # add attn-ed cls token to patch tokens
            new_patches = x[:, 1:, ...].clone() + x_cls.expand(-1, N-1, -1)
            x = torch.cat((x_cls, new_patches), dim=1)
        elif self.update == 'star':
            new_patches = x[:, 1:, ...].clone() * x_cls.expand(-1, N-1, -1)
            x = torch.cat((x_cls, new_patches), dim=1)
        elif self.update == 'cat':
            x = torch.cat((x_cls, x[:, 1:, ...].clone()), dim=1)
        if self.has_mlp:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


### ---------------------------------------------
# Model
### ---------------------------------------------

