import torch
import torch.nn as nn
# third-party dependencies
from timm.layers import DropPath, Mlp
# local dependencies
from .ir50_stage3 import iresnet50_stage3
from .cls_vit_stage3 import CLSAttention, NonMultiCLSBlock

class NonMultiCLSBlock_catAfterMlp(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
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
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_cls = x[:, 0:1, ...] + self.drop_path1(self.attn(self.norm1(x)))
        if self.has_mlp:
            x_cls = x_cls + self.drop_path2(self.mlp(self.norm2(x_cls)))
        if self.add_to_patch: # add attn-ed cls token to patches
            new_patches = x[:, 1:, ...].clone() + x_cls.expand(-1, N-1, -1)
            x = torch.cat((x_cls, new_patches), dim=1)
        else:
            x = torch.cat((x_cls, x[:, 1:, ...].clone()), dim=1)
        return x
    
class NonMultiCLSFER_stage3(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 embed_len: int=49,
                 embed_dim: int=512,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias=False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 block: nn.Module=NonMultiCLSBlock,
                add_to_patch: bool = False):
        super(NonMultiCLSFER_stage3, self).__init__()
        self.irback = iresnet50_stage3(num_features=num_classes)
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
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,   
                 add_to_patch=add_to_patch)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
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
        out = self.head(x_cls)
        return out
    
def get_NonMultiCLSFER_catAfterMlp(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            block=NonMultiCLSBlock_catAfterMlp)
    return model

def get_NonMutiCLSFER_addpatches(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,
                            block=NonMultiCLSBlock_catAfterMlp,
                            add_to_patch=True)
    return model