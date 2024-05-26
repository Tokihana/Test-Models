# inline
from typing import Optional
# third-party   
import torch 
import torch.nn as nn
from timm.models.vision_transformer import Block # original vit block
from timm.layers.helpers import make_divisible
# local  
from .ir50_14 import iresnet50
from .cat_cls_vit import NonMultiCLSBlock_catAfterMlp

class SE_block(nn.Module):
    def __init__(self, dim: int, mlp_ratio=4.):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, int(mlp_ratio*dim))
        self.relu = nn.GELU()
        self.linear2 = torch.nn.Linear(int(mlp_ratio*dim), dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x
    
class Conv_SE(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float=4.):
        super().__init__()
        self.fc1 = nn.Conv2d(dim ,int(mlp_ratio*dim), kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(int(mlp_ratio*dim))
        self.fc2 = nn.Conv2d(int(mlp_ratio*dim), dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        inputs = x
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

class CLSFER(nn.Module):
    def __init__(self,
                 img_size: int=112,
                 embed_len: int=196,
                 embed_dim: int=256,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias=False,
                 init_values: Optional[float]=None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 block: nn.Module=NonMultiCLSBlock_catAfterMlp,
                add_to_patch: bool = False,
                encoder_se: bool = False,   
                token_se: str='linear'):
        super(CLSFER, self).__init__()
        self.token_se = token_se
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        # stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        if encoder_se:
            self.blocks = nn.Sequential(*[])
            for i in range(depth):
                self.blocks.append(SE_block(channels=embed_dim))
                self.blocks.append(block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  init_values=init_values,
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,   
                 add_to_patch=add_to_patch))
            
        else:
            self.blocks = nn.Sequential(*[
                block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  init_values=init_values,
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,   
                 add_to_patch=add_to_patch)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if token_se == 'linear':
            self.token_se = SE_block(dim=embed_len)
        elif token_se == 'conv2d':
            self.token_se = Conv_SE(dim=embed_dim)
        self.headse = SE_block(dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
        
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        if self.token_se == 'conv2d':
            x_ir = self.token_se(x_ir)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        # squeeze channel dimension and scale tokens
        if self.token_se == 'linear':
            x_embed = self.tokense(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        # cat cls token, add pos embed
        x_cls = self.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed
        

        # attention blocks
        x = self.blocks(x) 
        x = self.norm(x)
        # output
        x_cls = x[:, 0:1, ...]
        # head
        x_cls = self.headse(x_cls)
        out = self.head(x_cls)
        return out.squeeze()

class Baseline_14(nn.Module):
    def __init__(self,
                 img_size: int=112,
                 embed_len: int=196,
                 embed_dim: int=256,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias=False,
                 qk_norm: bool = False,
                 init_values: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 norm_layer: nn.Module = nn.LayerNorm,
                encoder_se: bool=False,   
                token_se: str='linear'):
        super(Baseline_14, self).__init__()
        self.token_se = token_se
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        # stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        if encoder_se:
            self.blocks = nn.Sequential(*[])
            for i in range(depth):
                self.blocks.append(SE_block(channels=embed_dim))
                self.blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, init_values=init_values, 
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path))
        else:
            self.blocks = nn.Sequential(*[
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, init_values=init_values, 
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if token_se == 'linear':
            self.token_se = SE_block(dim=embed_len)
        elif token_se == 'conv2d':
            self.token_se = Conv_SE(dim=embed_dim)
        self.headse = SE_block(dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
            
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        if self.token_se == 'conv2d':
            x_ir = self.token_se(x_ir)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        # squeeze channel dimension and scale tokens
        if self.token_se == 'linear':
            x_embed = self.tokense(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        # cat cls token, add pos embed
        x_cls = self.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed
        

        # attention blocks
        x = self.blocks(x) 
        x = self.norm(x)
        # output
        x_cls = x[:, 0:1, ...]
        # head
        x_cls = self.headse(x_cls)
        out = self.head(x_cls)
        return out.squeeze()
    

def _get_14x14_CLSFER_baseline(config):
    model = Baseline_14(num_classes=config.MODEL.NUM_CLASS, 
                                depth=config.MODEL.DEPTH, 
                                mlp_ratio=config.MODEL.MLP_RATIO,
                                attn_drop=config.MODEL.ATTN_DROP,
                                qk_norm=config.MODEL.QK_NORM,
                                init_values=config.MODEL.LAYER_SCALE,   
                                token_se=config.MODEL.TOKEN_SE)
    return model
        
def _get_14x14_CLSFER_catAfterMlp(config):
    model = CLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=False,   
                                token_se=config.MODEL.TOKEN_SE)
    return model

def _get_14x14_CLSFER_addpatches(config):
    model = CLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=True,   
                                token_se=config.MODEL.TOKEN_SE)
    return model

def _get_14x14se_CLSFER_baseline(config):
    model = Baseline_14(num_classes=config.MODEL.NUM_CLASS, 
                                depth=config.MODEL.DEPTH, 
                                mlp_ratio=config.MODEL.MLP_RATIO,
                                attn_drop=config.MODEL.ATTN_DROP,
                                qk_norm=config.MODEL.QK_NORM,
                                init_values=config.MODEL.LAYER_SCALE,   
                       encoder_se=True,   
                                token_se=config.MODEL.TOKEN_SE)
    return model
        
def _get_14x14se_CLSFER_catAfterMlp(config):
    model = CLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=False,   
                            encoder_se=True,   
                                token_se=config.MODEL.TOKEN_SE)
    return model

def _get_14x14se_CLSFER_addpatches(config):
    model = CLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=True,
                            encoder_se=True,   
                                token_se=config.MODEL.TOKEN_SE)
    return model


def get_14x14_model(config):
    if config.MODEL.ARCH == '14x14_CLSFER_baseline':
        model = _get_14x14_CLSFER_baseline(config)
    elif config.MODEL.ARCH == '14x14_CLSFER_catAfterMlp':
        model = _get_14x14_CLSFER_catAfterMlp(config)
    elif config.MODEL.ARCH == '14x14_CLSFER_addpatches':
        model = _get_14x14_CLSFER_addpatches(config)
    elif config.MODEL.ARCH == '14x14se_CLSFER_baseline':
        model = _get_14x14se_CLSFER_baseline(config)
    elif config.MODEL.ARCH == '14x14se_CLSFER_catAfterMlp':
        model = _get_14x14se_CLSFER_catAfterMlp(config)
    elif config.MODEL.ARCH == '14x14se_CLSFER_addpatches':
        model = _get_14x14se_CLSFER_addpatches(config)
    return model