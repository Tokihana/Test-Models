# inline
from typing import Optional
# third party
import torch
import torch.nn as nn
# local
from .cat_cls_vit import NonMultiCLSBlock_catAfterMlp, SE_block
from .ir50_stage3 import iresnet50_stage3
from .mobilefacenet import MobileFaceNet

class CrossCLSBlock(nn.Module):
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
        super(CrossCLSBlock, self).__init__()
        self.img_block = NonMultiCLSBlock_catAfterMlp(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, init_values=init_values,
                                     attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, act_layer=act_layer, 
                                    norm_layer=norm_layer, has_mlp=has_mlp, add_to_patch=add_to_patch)
        self.lm_block = NonMultiCLSBlock_catAfterMlp(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, init_values=init_values,
                                     attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, act_layer=act_layer, 
                                    norm_layer=norm_layer, has_mlp=has_mlp, add_to_patch=add_to_patch)
    def forward(self, x):
        x_img, x_lm = x
        # cross cls token
        # [cls_lm,  patches_lm]  -> [cls_img, patches_lm]
        # [cls_img, patches_img] -> [cls_lm, patches_img]
        x_lm = torch.cat((x_img[:, 0:1, ...], x_lm[:, 1:, ...]), dim=1)
        x_img = torch.cat((x_lm[:, 0:1, ...], x_img[:, 1:, ...]), dim=1)
        # compute new cls token 
        x_lm = self.lm_block(x_lm)
        x_img = self.img_block(x_img)
        # cat new x
        x = [x_img, x_lm]
        return x
    
    
class CrossCLSFER(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 embed_len: int=49,
                 embed_dim: int=512,
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
                 block: nn.Module=CrossCLSBlock,
                add_to_patch: bool = False):
        super(CrossCLSFER, self).__init__()
        self.embed_len = embed_len
        # prepare irback
        self.irback = iresnet50_stage3(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        # prepare lm back
        self.lmback = MobileFaceNet([112, 112],136)
        lm_checkpoint = torch.load('./model/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        miss, unexcepted = self.lmback.load_state_dict(lm_checkpoint['state_dict'])
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del lm_checkpoint, miss, unexcepted
        
        # cls token, pos embed
        self.cls_token_ir = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_lm = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_ir = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        self.pos_embed_lm = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        # dpr
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        # cross cls blocks
        self.blocks = nn.Sequential(*[
            block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  init_values=init_values,
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,   
                 add_to_patch=add_to_patch)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.se = SE_block(input_dim=512)
        self.head = nn.Linear(embed_dim, num_classes) 
        
    def forward(self, x):
        '''
        x (B, 3, 112, 112)
        '''
        # prepare x_img, BCHW -> BNC
        x_ir = self.irback(x)
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        x_cls_ir = self.cls_token_ir.expand(x_ir.shape[0], -1, -1)
        x_img = torch.cat((x_cls_ir, x_embed), dim=1)
        x_img = x_img + self.pos_embed_ir
        
        # prepare x_lm
        _, x_lm = self.lmback(x)
        x_embed = x_ir.flatten(2).transpose(-2, -1)
        x_cls_lm = self.cls_token_lm.expand(x_lm.shape[0], -1, -1)
        x_lm = torch.cat((x_cls_lm, x_embed), dim=1)
        x_lm = x_lm + self.pos_embed_lm
        
        x = [x_img, x_lm]
        
        # forward blocks
        x = self.blocks(x)
        # forward head
        x_img, x_lm = x
        x_cls = x_img[:, 0:1, ...] + x_lm[:, 0:1, ...]
        x_cls = self.norm(x_cls)
        x_cls = self.se(x_cls)
        out = self.head(x_cls)
        return out
    
def get_CrossCLSFER(config):
    model = CrossCLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            init_values=config.MODEL.LAYER_SCALE,)
    return model