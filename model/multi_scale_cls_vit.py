# inline dependencies
from typing import Optional
# third-party dependencies
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
# local dependencies
from .cat_cls_vit import NonMultiCLSBlock_catAfterMlp, SE_block
from .ir50_stage3 import iresnet50_stage3
from .cls_vit_stage3 import CLSAttention, NonMultiCLSBlock

class MultiScaleBlock(nn.Module):
    def __init__(self, embed_len=49,
                 dims=[512, 256, 128], 
                 num_heads: int=8,   
                 mlp_ratio: float=4.,    
                 qkv_bias: float=0., 
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 drop_path: float=0.,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module=nn.LayerNorm,
                 has_mlp: bool=True,
                block: nn.Module=NonMultiCLSBlock_catAfterMlp):
        super(MultiScaleBlock, self).__init__()
        self.scale_num = len(dims)
        self.blocks = nn.ModuleList([block(dim=dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           attn_drop=attn_drop,
                                           proj_drop=proj_drop,
                                           drop_path=drop_path, 
                                           act_layer=act_layer, 
                                           norm_layer=norm_layer,
                                           has_mlp=has_mlp,) for dim in dims])
        self.upsample_m = nn.ConvTranspose1d(embed_len+1, embed_len+1, kernel_size=2, stride=2)
        self.upsample_s = nn.ConvTranspose1d(embed_len+1, embed_len+1, kernel_size=2, stride=2)
        
    def forward(self, x):
        '''x [x_l, x_m, x_s]
           x_l (BNC)
           x_m (BN(C/2))
           x_s (BN(C/4))'''
        x = [self.blocks[i](x[i]) for i in range(self.scale_num)]
        x[1] = self.upsample_s(x[2]) + x[1]
        x[0] = self.upsample_m(x[1]) + x[0]
        return x
    
class MultiScaleCLSFER(nn.Module):
    def __init__(self,
                 img_size: int=112,
                 embed_len: int=49,
                 embed_dim: int=512,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias=False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 init_values: Optional[float]=None,
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 block=NonMultiCLSBlock_catAfterMlp):
        super(MultiScaleCLSFER, self).__init__()
        self.irback = iresnet50_stage3(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        self.downsample_m = nn.Conv1d(embed_len+1, embed_len+1, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(embed_len+1, embed_len+1, kernel_size=4, stride=4)
        
        # stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.Sequential(*[
            MultiScaleBlock(dims=[embed_dim, embed_dim//2, embed_dim//4], embed_len=embed_len, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer, block=NonMultiCLSBlock_catAfterMlp)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.se = SE_block(input_dim=512)
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
        
        x = [x, self.downsample_m(x), self.downsample_s(x)]

        # attention blocks
        x = self.blocks(x) 
        x = self.norm(x[0])
        # output
        x_cls = x[:, 0, ...]
        # head
        x_cls = self.se(x_cls)
        out = self.head(x_cls)
        return out
    
def get_MultiScaleCLSFER(config):
    model = MultiScaleCLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,
                            block=NonMultiCLSBlock_catAfterMlp)
    return model