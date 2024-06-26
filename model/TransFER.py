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
from .star_attn import StarBlock

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

class LALayer(nn.Module):
    def __init__(
        self, 
        in_channel: int,
        reduction: int = 16):
        super(LALayer, self).__init__()
        self.spatial_atten = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // reduction, kernel_size = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // reduction, 1, kernel_size = 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        
        y = self.spatial_atten(x)
        return x * y

class CLSTransFER(nn.Module):
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
                encoder_se: bool = False,):
        super(CLSTransFER, self).__init__()
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
        self.la_layer = LALayer(in_channel=embed_dim)
        self.headse = SE_block(channels=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
        
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        x_ir = self.la_layer(x_ir)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
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

class TransFER(nn.Module):
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
                 block: nn.Module=Block,
                encoder_se: bool=False, ):
        super(TransFER, self).__init__()
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
                block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, init_values=init_values, 
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.la_layer = LALayer(in_channel=embed_dim)
        self.headse = SE_block(channels=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
            
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        x_ir = self.la_layer(x_ir)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
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
    

def _get_TransFER_baseline(config):
    model = TransFER(num_classes=config.MODEL.NUM_CLASS, 
                                depth=config.MODEL.DEPTH, 
                                mlp_ratio=config.MODEL.MLP_RATIO,
                                attn_drop=config.MODEL.ATTN_DROP,
                                qk_norm=config.MODEL.QK_NORM,
                                init_values=config.MODEL.LAYER_SCALE,)
    return model
        
def _get_TransFER_catAfterMlp(config):
    model = CLSTransFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=False,)
    return model

def _get_TransFER_addpatches(config):
    model = CLSTransFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=True,)
    return model

def _get_TransFER_Star(config):
    model = model = TransFER(num_classes=config.MODEL.NUM_CLASS, 
                                depth=config.MODEL.DEPTH, 
                                mlp_ratio=config.MODEL.MLP_RATIO,
                                attn_drop=config.MODEL.ATTN_DROP,
                                qk_norm=config.MODEL.QK_NORM,
                                init_values=config.MODEL.LAYER_SCALE,
                            block=StarBlock)
    return model


def get_TransFER(config):
    if config.MODEL.ARCH == 'TransFER':
        model = _get_TransFER_baseline(config)
    elif config.MODEL.ARCH == 'TransFER_catAfterMlp':
        model = _get_TransFER_catAfterMlp(config)
    elif config.MODEL.ARCH == 'TransFER_addpatches':
        model = _get_TransFER_addpatches(config)
    elif config.MODEL.ARCH == 'TransFER_Star':
        model = _get_TransFER_Star(config)
    return model