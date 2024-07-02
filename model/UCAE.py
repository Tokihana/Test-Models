# inline
import math
import unittest
from typing import Optional, List, Callable
# third-party
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block as ViTBlock
# local
from .CAE import Feature_Extractor, Linear_Mlp, CAEBlock, CrossCAEBlock, SEhead


### -----------------------
# Overall Architecture
### -----------------------
'''
Model:
IR50/MobileFaceNet              TransFormers
        |                        __________
        x1   -----------------  |   CAE_l  |
        |                       |     |    |
        x2      -----------     |   CAE_m  | 
        |                       |     |    |
        x3         ------       |   CAE_s  |
                                |        xN|
                                ------------
'''

### ----------------------
# Utils
### ----------------------
class Feature_Extractor(nn.Module):
    '''
    extract ir and lm features
    '''
    def __init__(self, ir_path = '', lm_path = ''):
        super().__init__()
        # ir backbone
        self.irback = iresnet50(num_features=7)
        if not ir_path == '':
            checkpoint = torch.load(ir_path)
            miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
            print(f'load IR50 check from {ir_path},\n Miss: {miss},\t Unexcept: {unexcepted}')
            del checkpoint, miss, unexcepted
        # lm backbone
        self.lmback = MobileFaceNet([112, 112], 136)
        if not lm_path == '':
            checkpoint = torch.load(lm_path, map_location=lambda storage, loc: storage)
            miss, unexcepted = self.lmback.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f'load MobileFaceNet check from {lm_path},\n Miss: {miss},\t Unexcept: {unexcepted}')
            del checkpoint, miss, unexcepted

        # lm projections
        dim1, dim2 = [64, 128], [128, 256]
        self.proj1 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim1[0], out_channels=dim1[1], kernel_size=1, padding=0, stride=1), # point-wise
            nn.Conv2d(in_channels=dim1[1], out_channels=dim1[1], kernel_size=3, padding=1, stride=1, groups=dim1[1]), # depth-wise
            nn.Conv2d(in_channels=dim1[1], out_channels=dim1[1], kernel_size=1, padding=0, stride=1), # point-wise
            nn.BatchNorm2d(dim1[1])
        ])
        self.proj2 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim2[0], out_channels=dim2[1], kernel_size=1, padding=0, stride=1), # point-wise
            nn.Conv2d(in_channels=dim2[1], out_channels=dim2[1], kernel_size=3, padding=1, stride=1, groups=dim2[1]), # depth-wise
            nn.Conv2d(in_channels=dim2[1], out_channels=dim2[1], kernel_size=1, padding=0, stride=1), # point-wise
            nn.BatchNorm2d(dim2[1])
        ])

    def forward(self, x):
        x_ir = self.irback(x)
        x_lm = self.lmback(x)
        # project x_lm
        x1, x2, x3 = x_lm
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        x_lm = [x1, x2, x3]

        return x_ir, x_lm

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution, input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat

class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x

### -----------------------
# Block
### -----------------------
class UBlock(nn.Module):
    def __init__(self, fig_size: List[int]=[28, 14, 7],
                 dims: List[int]=[128, 256, 512],
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
                 block: nn.Module=CAEBlock,
                 ):
        super().__init__()
        self.block_l = CrossCAEBlock(dim=dims[0], 
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
                      block=block)
        self.block_m = CrossCAEBlock(dim=dims[1], 
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
                      block=block)
        self.block_s = CrossCAEBlock(dim=dims[2], 
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
                      block=block)
        self.expand_s = PatchExpand(input_resolution=fig_size[2], dim=dims[2], dim_scale=2)
        self.expand_m = PatchExpand(input_resolution=fig_size[1], dim=dims[1], dim_scale=2) 
        self.proj_s = nn.Sequential(*[norm_layer(dims[2]), act_layer(), nn.Linear(dims[2], dims[1])])
        self.proj_m = nn.Sequential(*[norm_layer(dims[1]), act_layer(), nn.Linear(dims[1], dims[0])])
        

    def forward(self, x):
        # process block
        x_l, x_m, x_s = x
        x_s = self.block_s(x_s)
        x_m = self.block_m(x_m)
        x_l = self.block_l(x_l)

        '''
        # upscale x_s, consists of two branch: lm and ir
        #cls_m = [m[:, 0:1, ...] + self.proj_s(s[:, 0:1, ...]) for m, s in zip(x_m, x_s)]
        cls_m = [m[:, 0:1, ...] for m in x_m]
        patch_m = [m[:, 1:, ...] + self.expand_s(s[:, 1:, ...]) for m, s in zip(x_m, x_s)]
        x_m = [torch.cat((cls, m), dim=1) for cls, m in zip(cls_m, patch_m)]
        # upscale x_m
        #cls_l = [l[:, 0:1, ...] + self.proj_m(m[:, 0:1, ...]) for l, m in zip(x_l, x_m)]
        cls_l = [l[:, 0:1, ...] for l in x_l]
        patch_l = [l[:, 1:, ...] + self.expand_m(m[:, 1:,...]) for l, m in zip(x_l, x_m)]
        x_l = [torch.cat((cls, l), dim=1) for cls, l in zip(cls_l, patch_l)]
        '''
        x = [x_l, x_m, x_s]
        return x
        

### -----------------------
# Models
### -----------------------
class UCrossModel(nn.Module):
    def __init__(self, img_size: int=112,
                embed_len: List[int]=[784, 196, 49],
                embed_dim: List[int]=[128, 256, 512],
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
                block: nn.Module=CAEBlock,
                mlp_layer: nn.Module=Linear_Mlp,):
        super().__init__()
        self.feature_extractor = Feature_Extractor(
            ir_path = './model/pretrain/ir50_backbone.pth',
            lm_path = './model/pretrain/mobilefacenet_model_best.pth.tar'
        )
        self.cls_token_ir = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim)) for dim in embed_dim])
        self.cls_token_lm = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim)) for dim in embed_dim])
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, l + 1, dim)) for l, dim in zip(embed_len, embed_dim)])

        self.blocks = nn.Sequential(*[
                     UBlock(fig_size=[28, 14, 7],
                            dims=embed_dim,
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
                            mlp_layer=mlp_layer,
                            block=block,
                            ) for i in range(depth)])

        self.norms = nn.ModuleList([nn.ModuleList([norm_layer(dim) for i in range(2)]) for dim in embed_dim])
        self.heads = nn.ModuleList([nn.ModuleList([SEhead(dim=dim, num_classes=num_classes, head_drop=head_drop) for i in range(2)]) for dim in embed_dim])
        
    def forward(self, x):
        x_irs, x_lms = self.feature_extractor(x)

        ir_embed = [x.flatten(2).transpose(-2, -1) for x in x_irs]
        lm_embed = [x.flatten(2).transpose(-2, -1) for x in x_lms]
        x_irs = [torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1) for cls, x in zip(self.cls_token_ir, ir_embed)]
        x_lms = [torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1) for cls, x in zip(self.cls_token_lm, lm_embed)]
        x_irs = [x + pos for x, pos in zip(x_irs, self.pos_embed)]

        x = [[x_ir, x_lm] for x_ir, x_lm in zip(x_irs, x_lms)]

        x = self.blocks(x)

        x = [[norm(x) for norm, x in zip(self.norms[i], x[i])] for i in range(3)]

        out = []
        for i in range(3):
            for j in range(2):
                cls = x[i][j][:, 0:1, ...]
                out.append(self.heads[i][j](cls))
        out = torch.mean(torch.stack(out, dim=0), dim=0)
        return out.squeeze()


class SingleUCrossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Feature_Extractor(
            ir_path = './model/pretrain/ir50_backbone.pth',
            lm_path = './model/pretrain/mobilefacenet_model_best.pth.tar'
        )

        self.cross_supervise = nn.ModuleList([
            block()
            for i in range(3)
        ])

        self.blocks = nn.Sequential(*[
            block()
            for i in range(depth)
        ])

        self.norm = norm_layer(dim)
        self.head = nn.Sequential(*[
            SE_block(input_dim=512),
            ClassificationHead(input_dim=512, target_dim=7)
        ])
            
        
        
    def forward_features(self, x):
        x_irs, x_lms = self.feature_extractor(x)
        x = [x_ir * x_lm for x_ir, x_lm in zip(x_irs, x_lms)]

        ir_embed = [x.flatten(2).transpose(-2, -1) for x in x_irs]
        lm_embed = [x.flatten(2).transpose(-2, -1) for x in x_lms]
        x_irs = [torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1) for cls, x in zip(self.cls_token_ir, ir_embed)]
        x_lms = [torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1) for cls, x in zip(self.cls_token_lm, lm_embed)]
        x_irs = [x + pos for x, pos in zip(x_irs, self.pos_embed)]

        
        return x
        
    def forward_head(self, x):
        x = self.head(x)
        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

### ---------------------------------------------
# create_model
### ---------------------------------------------
def get_UCrossCAE(config):
    if config.MODEL.ARCH == 'UCrossAC-CAE':
        return _get_UCrossCAE(config, block=CAEBlock)

    if config.MODEL.ARCH == 'UCrossViT':
        return _get_UCrossCAE(config, block=ViTBlock)
    
def _get_UCrossCAE(config, block):
    model = UCrossModel(img_size=config.DATA.IMG_SIZE,
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
                        block=block)
    return model


### --------------------------
# Unit Tests
### --------------------------
class UtilTests(unittest.TestCase):
    def test_PatchExpand(self):
        '''
        given input patches shaped B, N+1, C,
        Expand it to B, N*4+1, C//2
        '''
        B, N, C = 5, 50, 512
        H, W = 7, 7
        model = PatchExpand(input_resolution=H, dim=C, dim_scale=2)
        x = torch.rand((B, N, C))
        out = model(x[:, 1:, ...])
        print(out.shape)
        self.assertEqual(out.shape, torch.Size([B, (H*W*4), C//2]))
        
    def test_ClassifierHead(self):
        B, N, C, num_c = 5, 50, 512, 7
        head = nn.Sequential(*[
            SE_block(input_dim=512),
            ClassificationHead(input_dim=512, target_dim=7)
        ])
        x = torch.rand((B, C))
        out = head(x)
        print(out.shape)

class BlockTests(unittest.TestCase):
    def test_UBlock(self):
        B = 5
        Ns = [785, 197, 50]
        Cs = [128, 256, 512]
        size = [28, 14, 7]
        model = UBlock(fig_size=size, dims=Cs)
        x = []
        for i in range(3):
            x.append([torch.rand((B, Ns[i], Cs[i])) for j in range(2)])

        out = model(x)
        for i in range(3):
            for j in range(2):
                print(out[i][j].shape)
                self.assertEqual(out[i][j].shape, torch.Size([B, Ns[i], Cs[i]]))


class ModelTests(unittest.TestCase):
    def test_UModel(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        imgs = torch.rand((B, C, H, W))
        model = UCrossModel(num_classes=num_class)
        out = model(imgs)
        print(out.shape)

    def test_UModel_with_ViTBlock(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        imgs = torch.rand((B, C, H, W))
        model = UCrossModel(num_classes=num_class, block=ViTBlock)
        out = model(imgs)
        print(out.shape)
                


if __name__ == '__main__':
    unittest.main()