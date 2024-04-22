from typing import Optional
# third-party dependencies
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
# local dependencies
from .ir50_stage3 import iresnet50_stage3
from .cls_vit_stage3 import CLSAttention, NonMultiCLSBlock
from .cat_cls_vit import NonMultiCLSFER_stage3

class CONV_PROJ_CLSAttention(nn.Module):
    def __init__(self, dim: int, 
                num_heads: int=8,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,):
        super(CONV_PROJ_CLSAttention, self).__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5
        
        # self.wq, self.wk, self.wv = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias)
        self.wq, self.wk, self.wv = nn.Conv1d(dim, dim, kernel_size=1, stride=1), nn.Conv1d(dim, dim, kernel_size=1, stride=1), nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x (B, N, C) -> return (B, 1, C)'''
        B, N, C = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...].permute(0, 2, 1)).permute(0, 2, 1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k =  self.wk(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # attn: BH1(C/H) @ BH(C/H)N -> BH1N
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = attn.softmax(dim=-1) # dim that will compute softmax, every slice along this dim will sum to 1, for attn case, is N
        attn = self.attn_drop(attn)
        
        # BH1N @ BHN(C/H) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class NonMultiCLSBlock_catAfterMlp(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
                init_values: Optional[float] = None,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                has_mlp: bool=True,
                add_to_patch: bool=False):
        super(NonMultiCLSBlock_catAfterMlp, self).__init__()
        self.has_mlp = has_mlp
        self.add_to_patch = add_to_patch
        self.norm1 = norm_layer(dim)
        self.attn = CONV_PROJ_CLSAttention(dim, 
                                 num_heads=num_heads,
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
        x_cls = x[:, 0:1, ...] + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.has_mlp:
            x_cls = x_cls + self.drop_path2(self.ls2(self.mlp(self.norm2(x_cls))))
        if self.add_to_patch: # add attn-ed cls token to patches
            new_patches = x[:, 1:, ...].clone() + x_cls.expand(-1, N-1, -1)
            x = torch.cat((x_cls, new_patches), dim=1)
        else:
            x = torch.cat((x_cls, x[:, 1:, ...].clone()), dim=1)
        return x
    

def get_NonMutiCLSFER_conv_proj(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=False)
    return model