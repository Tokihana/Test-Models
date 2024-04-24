# inline
from typing import Optional
# third-party   
import torch 
import torch.nn as nn
from timm.models.vision_transformer import Block # original vit block
# local  
from .ir50_14 import iresnet50
from .cat_cls_vit import NonMultiCLSBlock_catAfterMlp, SE_block

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
                add_to_patch: bool = False):
        super(CLSFER, self).__init__()
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        # stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.Sequential(*[
            block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  init_values=init_values,
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,   
                 add_to_patch=add_to_patch)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.se = SE_block(input_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
        
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        # cat cls token, add pos embed
        x_cls = self.cls_token.expand(x_ir.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed

        # attention blocks
        x = self.blocks(x) 
        x = self.norm(x)
        # output
        x_cls = x[:, 0, ...]
        # head
        x_cls = self.se(x_cls)
        out = self.head(x_cls)
        return out

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
                 norm_layer: nn.Module = nn.LayerNorm):
        super(Baseline_14, self).__init__()
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, init_values=init_values, 
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.se = SE_block(input_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) 
            
    def forward(self, x):
        # get features
        x_ir = self.irback(x)
        # patchify, BCHW -> BNC
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        # cat cls token, add pos embed
        x_cls = self.cls_token.expand(x_ir.shape[0], -1, -1)
        x = torch.cat((x_cls, x_embed), dim=1)
        x = x + self.pos_embed

        # attention blocks
        x = self.blocks(x) 
        x = self.norm(x)
        # output
        x_cls = x[:, 0, ...]
        # head
        x_cls = self.se(x_cls)
        out = self.head(x_cls)
        return out
    
def _get_14x14_CLSFER_baseline(config):
    model = Baseline_14(num_classes=config.MODEL.NUM_CLASS, 
                                depth=config.MODEL.DEPTH, 
                                mlp_ratio=config.MODEL.MLP_RATIO,
                                attn_drop=config.MODEL.ATTN_DROP,
                                qk_norm=config.MODEL.QK_NORM,
                                init_values=config.MODEL.LAYER_SCALE)
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
                            add_to_patch=False)
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
                            add_to_patch=True)
    return model

def get_14x14_model(config):
    if config.MODEL.ARCH == '14x14_CLSFER_baseline':
        model = _get_14x14_CLSFER_baseline(config)
    elif config.MODEL.ARCH == '14x14_CLSFER_catAfterMlp':
        model = _get_14x14_CLSFER_catAfterMlp(config)
    elif config.MODEL.ARCH == '14x14_CLSFER_addpatches':
        model = _get_14x14_CLSFER_addpatches(config)
    return model