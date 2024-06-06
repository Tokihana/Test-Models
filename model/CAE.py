# inline dependencies
import math
import unittest
from typing import Optional, List
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

### ------------------------
# utils
### ------------------------
class Conv1d_FC(nn.Module):
    '''
    input: 
        x (B, N, C)
    '''
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int=1, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, with_norm: bool=True):
        super().__init__()
        self.conv = nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if with_norm:
            self.bn = nn.BatchNorm1d(out_chans)
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0) 
        else:
            self.bn = nn.Identity()
            
    def forward(self, x):
        x = self.bn(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        return x
    
class Linear_FC(nn.Module):
    '''
    linear full connection layer with LayerNorm
    '''
    def __init__(self, in_chans: int, out_chans: int, with_norm: bool=True):
        super().__init__()
        self.fc = nn.Linear(in_chans, out_chans,)
        self.norm = nn.LayerNorm(out_chans) if with_norm else nn.Identity()
        
    def forward(self, x):
        x = self.norm(self.fc(x))
        return x
        
    
class SE_block(nn.Module):
    def __init__(self, dim: int, rd_ratio=1./16):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, int(rd_ratio*dim))
        self.relu = nn.GELU()
        self.linear2 = torch.nn.Linear(int(rd_ratio*dim), dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = x.mean(dim=1, keepdim=True)
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x
    
class Conv1d_SE(nn.Module):
    def __init__(self, dim: int, rd_ratio=1./16):
        super().__init__()
        rd_chans = int(rd_ratio * dim)
        self.fc1 = nn.Conv1d(dim, rd_chans, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(rd_chans)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv1d(rd_chans, dim, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = x.mean(dim=1, keepdim=True).permute(0, 2, 1)
        x1 = self.sigmoid(self.fc2(self.act_layer(self.bn(self.fc1(x1)))))
        x = x * x1.permute(0, 2, 1)
        return x
    
    
class Conv_SE(nn.Module):
    def __init__(self, dim: int, rd_ratio: float=1./16):
        super().__init__()
        self.fc1 = nn.Conv2d(dim ,int(rd_ratio*dim), kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(int(rd_ratio*dim))
        self.fc2 = nn.Conv2d(int(mlp_ratio*dim), dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        inputs = x
        x = x.mean(dim = 1, keepdim=True)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x * inputs
        return x
            
class Conv1d_Mlp(nn.Module):
    '''
    1d convlution mlp, input x (B, N, C)
    '''
    def __init__(self, dim: int, 
                 mlp_ratio: float=4.,
                 drop: float=0.,
                 act_layer: nn.Module=nn.GELU,):
        super().__init__()
        self.fc1 = nn.Conv1d(dim, int(dim*mlp_ratio), kernel_size=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(int(dim*mlp_ratio))
        self.fc2 = nn.Conv1d(int(dim*mlp_ratio), dim, kernel_size=1)
        self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 1))
        x = self.act(x)
        x = self.drop1(x)
        x = self.bn(x)
        x = self.fc2(x)
        x = self.drop2(x).permute(0, 2, 1)
        return x
    
class Linear_Mlp(nn.Module):
    '''
    linear mlp, input x (B, N, C)
    '''
    def __init__(self, dim: int,
                 mlp_ratio: float=4.,
                 drop: float=0.,
                 act_layer: nn.Module=nn.GELU,):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim*mlp_ratio))
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(int(dim*mlp_ratio))
        self.fc2 = nn.Linear(int(dim*mlp_ratio), dim)
        self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class SEhead(nn.Sequential):
    def __init__(self, dim: int, num_classes: int, head_norm: bool=True, head_se: bool=True, head_drop: float=0.):
        super().__init__()
        if head_norm:
            self.add_module('head_norm', nn.LayerNorm(dim))
        if head_se:
            self.add_module('se_block', SE_block(dim=dim))
        if head_drop > 0.:
            self.add_module('head_drop', nn.Dropout(head_drop))
        self.add_module('linear', nn.Linear(dim, num_classes))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_chans:int, out_chans:int, scale_factor:int=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        self.conv = DoubleConv(in_channels=in_chans, out_channels=out_chans)
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        # BNC -> BCN -> BCHW
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.up(x)
        x = self.conv(x)
        x = x.flatten(2).transpose(-2, -1)
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
        attn = q @ k.transpose(-2, -1) * self.scale
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
class CAEBlock(nn.Module):
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
                mlp_layer: nn.Module=Linear_Mlp,
                ):
        super().__init__()
        # CAE
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
        
        # mlp 
        self.fc = Linear_FC(in_chans=dim, out_chans=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        '''
        x (B, N, C)
        '''
        B, N, C = x.shape
        inputs = x
        # BNC -> B1C
        x_cls = x[:, 0:1, ...] + self.drop_path1(self.ls1(self.attn(self.norm1(x))))    
            
        # mlp
        x_cls = x_cls + self.drop_path2(self.ls2(self.mlp(self.norm2(x_cls))))
        new_patches = self.fc(x[:, 1:, ...]) + x_cls
        x = torch.cat((x_cls, new_patches), dim=1)
            
        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, dims: List[int] = [128, 256, 512],
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
                mlp_layer: nn.Module=Conv1d_Mlp,
                block: nn.Module=CAEBlock, 
                ):
        super().__init__()
        if block == CAEBlock:
            self.multi_scale_blocks = nn.ModuleList([
                block(dim=dim, 
                     num_heads=num_heads, 
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_norm=qk_norm,
                     init_values=init_values,
                     attn_drop=attn_drop,
                     proj_drop=proj_drop,
                     drop_path=drop_path,
                     act_layer=act_layer,
                     norm_layer=norm_layer,
                     mlp_layer=mlp_layer,)
            for dim in dims])
        elif block == ViTBlock:
            self.multi_scale_blocks = nn.ModuleList([
                block(dim=dim, 
                      num_heads=num_heads, 
                      mlp_ratio=mlp_ratio, 
                      qkv_bias=qkv_bias,
                      qk_norm=qk_norm, 
                      init_values=init_values, 
                      attn_drop=attn_drop, 
                      proj_drop=proj_drop, 
                      drop_path=drop_path,
                      act_layer=act_layer,
                      norm_layer=norm_layer,)
                for dim in dims])
        self.upsample_s = Up(in_chans=dims[2], out_chans=dims[1], scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_m = Up(in_chans=dims[1], out_chans=dims[0], scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, xs):
        '''
        given xs List[Tensor]: list of token sequence in different scale
        '''
        N = len(xs)
        x1, x2, x3 = [self.multi_scale_blocks[i](xs[i]) for i in range(N)]
        x2 = x2 + self.upsample_s(x3)
        x1 = x1 + self.upsample_m(x2)
        xs = [x1, x2, x3]
        return xs


### ---------------------------------------------
# Model
### ---------------------------------------------

class SingleModel(nn.Module):
    '''
    use 14x14 feature maps from IR50, fed to {depth} layer blocks
    '''
    def __init__(self, 
                img_size: int=112,
                embed_len: int=196,
                embed_dim: int=256,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias=False,
                qk_norm=True,
                init_values: Optional[float]=None,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                head_drop: float=0.,
                depth: int=8,
                num_classes: int=7,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                block: nn.Module=ViTBlock,
                mlp_layer: nn.Module=Linear_Mlp,
                token_se: str='linear'):
        super().__init__()
        self.token_se = token_se
        # ir50 backbone
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        if token_se == 'linear':
            self.token_se_block = SE_block(dim=embed_len)
        elif token_se == 'conv2d':
            self.token_se_block = Conv_SE(dim=embed_dim)
        elif token_se == 'conv1d':
            self.token_se_block = Conv1d_SE(dim=embed_len)
        
        # running blocks
        ## cls token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        ## stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        ## blocks 
        if block == CAEBlock: # using star block
            self.blocks = nn.Sequential(*[
                block(dim=embed_dim, 
                      num_heads=num_heads, 
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_norm=qk_norm,
                      init_values=init_values,
                      attn_drop=attn_drop,
                      proj_drop=proj_drop,
                      drop_path=dpr[i],
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      mlp_layer=mlp_layer,)
            for i in range(depth)])
        elif block == ViTBlock:
            self.blocks = nn.Sequential(*[
                block(dim=embed_dim, 
                      num_heads=num_heads, 
                      mlp_ratio=mlp_ratio, 
                      qkv_bias=qkv_bias,
                      qk_norm=qk_norm, 
                      init_values=init_values, 
                      attn_drop=attn_drop, 
                      proj_drop=proj_drop, 
                      drop_path=dpr[i],
                      act_layer=act_layer,
                      norm_layer=norm_layer,)
                for i in range(depth)])
        else:
            raise ModuleNotFoundError(f'unsupported block type {block}')
        
        self.norm = norm_layer(embed_dim)
        self.head = SEhead(dim=embed_dim, num_classes=num_classes, head_drop=head_drop)
        
    def forward(self, x):
        # get stem output
        _, _, x_ir = self.irback(x)
        if self.token_se == 'conv2d':
            x_ir = self.token_se_block(x_ir)
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        if self.token_se in ['linear', 'conv1d']:
            x_embed = self.token_se_block(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        x_cls = self.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed
        
        # process blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # output
        x_cls = x[:, 0:1, ...]
        ## classification head
        out = self.head(x_cls)
        return out.squeeze()


class SingleStyleMultiScaleModel(nn.Module):
    '''
    simply concat the features at different scale and process the concatenated tokens in a single model
    '''
    def __init__(self, 
                img_size: int=112,
                embed_len: int=196,
                embed_dim: int=768,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias=False,
                qk_norm=True,
                init_values: Optional[float]=None,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                head_drop: float=0.,
                depth: int=8,
                num_classes: int=7,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                block: nn.Module=ViTBlock,
                mlp_layer: nn.Module=Linear_Mlp,
                token_se: str='linear'):
        super().__init__()
        self.token_se = token_se
        # ir50 backbone
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted

        o = self.irback(torch.zeros((1, 3, img_size, img_size)))
        dims = [o[0].shape[1], o[1].shape[1], o[2].shape[1]]
        self.embed1 = nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(dims[1], embed_dim, kernel_size=3, stride=2, padding=1),)
        self.embed2 = nn.Conv2d(dims[1], embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed3 = nn.Conv2d(dims[2], embed_dim, kernel_size=1)
        
        if token_se == 'linear':
            self.token_se_block = SE_block(dim=embed_len)
        elif token_se == 'conv2d':
            self.token_se_block = Conv_SE(dim=embed_dim)
        elif token_se == 'conv1d':
            self.token_se_block = Conv1d_SE(dim=embed_len)
        
        # running blocks
        ## cls token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len*3 + 1, embed_dim))
        ## stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        ## blocks 
        if block == CAEBlock: # using star block
            self.blocks = nn.Sequential(*[
                block(dim=embed_dim, 
                      num_heads=num_heads, 
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_norm=qk_norm,
                      init_values=init_values,
                      attn_drop=attn_drop,
                      proj_drop=proj_drop,
                      drop_path=dpr[i],
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      mlp_layer=mlp_layer,)
            for i in range(depth)])
        elif block == ViTBlock:
            self.blocks = nn.Sequential(*[
                block(dim=embed_dim, 
                      num_heads=num_heads, 
                      mlp_ratio=mlp_ratio, 
                      qkv_bias=qkv_bias,
                      qk_norm=qk_norm, 
                      init_values=init_values, 
                      attn_drop=attn_drop, 
                      proj_drop=proj_drop, 
                      drop_path=dpr[i],
                      act_layer=act_layer,
                      norm_layer=norm_layer,)
                for i in range(depth)])
        else:
            raise ModuleNotFoundError(f'unsupported block type {block}')
        
        self.norm = norm_layer(embed_dim)
        self.head = SEhead(dim=embed_dim, num_classes=num_classes, head_drop=head_drop)
        
    def forward(self, x):
        # get stem output
        x1, x2, x3 = self.irback(x)
        x1, x2, x3 = self.embed1(x1).flatten(2).transpose(-2, -1), self.embed2(x2).flatten(2).transpose(-2, -1), self.embed3(x3).flatten(2).transpose(-2, -1)
        
        x_embed = torch.cat([x1, x2, x3], dim=1)
        x_cls = self.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed
        
        # process blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # output
        x_cls = x[:, 0:1, ...]
        ## classification head
        out = self.head(x_cls)
        return out.squeeze()
    

    
### ---------------------------------------------
# create_model
### ---------------------------------------------
def get_AC_CAE(config):
    if config.MODEL.ARCH == 'AC-CAE_single':
        return _get_single_CAE(config)
    elif config.MODEL.ARCH == 'baseline_single':
        return _get_single_baseline(config)
    elif config.MODEL.ARCH == 'SingleStyle_multiCAE':
        return _get_single_style_multi_CAE(config)
    elif config.MODEL.ARCH == 'SingleStyle_multiBaseline':
        return _get_single_style_multi_baseline(config)

def _get_single_baseline(config):
    if config.MODEL.MLP_LAYER == 'linear':
        mlp_layer = Linear_Mlp
    elif config.MODEL.MLP_LAYER == 'conv1d':
        mlp_layer = Conv1d_Mlp
    else:
        raise ValueError(f'not support mlp layer type "{mlp_layer}"')
    model = SingleModel(img_size=config.DATA.IMG_SIZE,
                        mlp_ratio=config.MODEL.MLP_RATIO, 
                        num_classes=config.MODEL.NUM_CLASS,
                        qkv_bias=config.MODEL.QKV_BIAS,
                        qk_norm=config.MODEL.QK_NORM,
                        init_values=config.MODEL.LAYER_SCALE,
                        attn_drop=config.MODEL.ATTN_DROP,
                        proj_drop=config.MODEL.PROJ_DROP,
                        drop_path=config.MODEL.DROP_PATH,
                        head_drop=config.MODEL.HEAD_DROP,
                        depth=config.MODEL.DEPTH,
                        block=ViTBlock,
                        mlp_layer=mlp_layer,)
    return model

def _get_single_CAE(config):
    if config.MODEL.MLP_LAYER == 'linear':
        mlp_layer = Linear_Mlp
    elif config.MODEL.MLP_LAYER == 'conv1d':
        mlp_layer = Conv1d_Mlp
    else:
        raise ValueError(f'not support mlp layer type "{mlp_layer}"')
    model = SingleModel(img_size=config.DATA.IMG_SIZE,
                        mlp_ratio=config.MODEL.MLP_RATIO, 
                        num_classes=config.MODEL.NUM_CLASS,
                        qkv_bias=config.MODEL.QKV_BIAS,
                        qk_norm=config.MODEL.QK_NORM,
                        init_values=config.MODEL.LAYER_SCALE,
                        attn_drop=config.MODEL.ATTN_DROP,
                        proj_drop=config.MODEL.PROJ_DROP,
                        drop_path=config.MODEL.DROP_PATH,
                        head_drop=config.MODEL.HEAD_DROP,
                        depth=config.MODEL.DEPTH,
                        block=CAEBlock,
                        mlp_layer=mlp_layer,)
    return model

def _get_single_style_multi_CAE(config):
    model = SingleStyleMultiScaleModel(img_size=config.DATA.IMG_SIZE,
                        mlp_ratio=config.MODEL.MLP_RATIO, 
                        num_classes=config.MODEL.NUM_CLASS,
                        qkv_bias=config.MODEL.QKV_BIAS,
                        qk_norm=config.MODEL.QK_NORM,
                        init_values=config.MODEL.LAYER_SCALE,
                        attn_drop=config.MODEL.ATTN_DROP,
                        proj_drop=config.MODEL.PROJ_DROP,
                        drop_path=config.MODEL.DROP_PATH,
                        head_drop=config.MODEL.HEAD_DROP,
                        depth=config.MODEL.DEPTH,
                        block=CAEBlock,)
    return model

def _get_single_style_multi_baseline(config):
    model = SingleStyleMultiScaleModel(img_size=config.DATA.IMG_SIZE,
                        mlp_ratio=config.MODEL.MLP_RATIO, 
                        num_classes=config.MODEL.NUM_CLASS,
                        qkv_bias=config.MODEL.QKV_BIAS,
                        qk_norm=config.MODEL.QK_NORM,
                        init_values=config.MODEL.LAYER_SCALE,
                        attn_drop=config.MODEL.ATTN_DROP,
                        proj_drop=config.MODEL.PROJ_DROP,
                        drop_path=config.MODEL.DROP_PATH,
                        head_drop=config.MODEL.HEAD_DROP,
                        depth=config.MODEL.DEPTH,
                        block=ViTBlock,)
    return model
    
### ---------------------------------------------
# unittest
### ---------------------------------------------
class UtilTests(unittest.TestCase):
    def test_Double_Conv(self):
        B, C, H, W = 5, 512, 7, 7
        x = torch.rand((B, C, H, W))
        layer = DoubleConv(in_channels=C, out_channels=C)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([B, C, H, W]))
        
    def test_Up(self):
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        layer = Up(in_chans=C, out_chans=C//2)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([B, 196, C//2]))

class ClassAttentionTests(unittest.TestCase):
    '''
    test class attention computation.
    given inputs shaped (B, N, C),
    suppose outputs shaped (B, 1, C)
    '''
    def test_class_attn(self):
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        attn_layer = CLSAttention(dim=C, num_heads=8, qkv_bias=False, qk_norm=True,)
        out = attn_layer(x)
        self.assertEqual(out.shape, torch.Size([B, 1, C]))
        
class BlockTests(unittest.TestCase):
    def test_CAEBlock(self):
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        block = CAEBlock(dim=C)
        self.assertEqual(block(x).shape, torch.Size([B, N, C]))

    def test_MultiScaleBlock(self):
        B = 5
        Ns = [28*28, 14*14, 7*7]
        Cs = [128, 256, 512]
        xs = [torch.rand((B, Ns[i], Cs[i])) for i in range(3)]

        block = MultiScaleBlock(dims=Cs)
        out = block(xs)
        for i in range(3):
            self.assertEqual(out[i].shape, torch.Size([B, Ns[i], Cs[i]]))

        block = MultiScaleBlock(dims=Cs, block=ViTBlock)
        out = block(xs)
        for i in range(3):
            self.assertEqual(out[i].shape, torch.Size([B, Ns[i], Cs[i]]))

class ModelTests(unittest.TestCase):
    def test_baseline(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        images = torch.rand((B, C, H, W))
        model = SingleModel(num_classes=num_class, block=ViTBlock)
        out = model(images)
        self.assertEqual(out.shape, torch.Size([B, num_class]))
        
    def test_CAE(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        images = torch.rand((B, C, H, W))
        model = SingleModel(num_classes=num_class, block=CAEBlock)
        out = model(images)
        self.assertEqual(out.shape, torch.Size([B, num_class]))

    def test_SingleStyleMultiScaleModel(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        images = torch.rand((B, C, H, W))
        # TEST CAEBlock
        model = SingleStyleMultiScaleModel(num_classes=num_class, block=CAEBlock)
        out = model(images)
        self.assertEqual(out.shape, torch.Size([B, num_class]))
        # TEST ViTBlock
        model = SingleStyleMultiScaleModel(num_classes=num_class, block=CAEBlock)
        out = model(images)
        self.assertEqual(out.shape, torch.Size([B, num_class]))
    
if __name__ == '__main__':
    unittest.main()