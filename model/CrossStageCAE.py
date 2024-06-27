# inline
import math
import unittest
from typing import Optional, List, Callable
# third-party
import torch
import torch.nn as nn
# local
from .CAE import Feature_Extractor, Linear_Mlp, CAEBlock, CrossCAEBlock, SEhead

### -----------------------
# Overall Arch
### -----------------------
'''
Swin-Style Multi-Stage Cross AC-CAE
Model:
    Feature_Extractor (Stem):
        IR50
        MobileFaceNet
    Stages:
        StageBlock:
            (no downsample as i == 0)
            CrossCAEBlocks
        ...
        StageBlock:
            Transition (downsample)
            CrossCAEBlocks
        StageBlock:
            Transition (downsample)
            CrossCAEBlocks
    ClassiferHead
'''

### -----------------------
# Utils
### -----------------------
class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % 2 == 0, f"x height ({H}) is not even." 
        assert W % 2 == 0, f"x width ({W}) is not even."
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x

### ----------------------
# Blocks
### ----------------------
class StageBlock(nn.Module):
    '''
    Init:
        
    Forward:
        x (B, N+1, C) in xs
    Return:
        x (B, N2+1, C2) in xs
    '''
    def __init__(self, dim: int,
                out_dim: int, 
                num_heads: int=8,
                downsample: bool = True, # not downsample at stage 1
                depth: int=2, 
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
                block: nn.Module=CAEBlock, ):
        super().__init__()
        self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer) if downsample else nn.Identity()
        self.proj_cls_token = nn.Linear(dim, out_dim) if downsample else nn.Identity()
        blk_dim = out_dim if downsample else dim
        self.blocks = nn.Sequential(*[
            block(dim=blk_dim, 
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
            for i in range(depth)
        ])

    def forward(self, x):
        # B(N+1)C > x_cls (B1C), x (BHWC)
        B, N, C = x.shape
        x_cls = x[:, 0:1, ...]
        H = W = int(math.sqrt(N-1))
        x = x[:, 1:, ...].reshape(B, H, W, C)
        # downsample
        x = self.downsample(x)
        x_cls = self.proj_cls_token(x_cls)
        # blocks
        B, H, W, C = x.shape # new B, H, W, C
        x = x.reshape(B, -1, C)
        x = torch.cat((x_cls, x), dim=1)
        x = self.blocks(x)
        return x



class CrossStageBlock(nn.Module):
    '''
    Init:
        
    Forward:
        x (B, H, W, C) in xs
    Return:
        x (B, H//2, W//2, C*2) in xs
    '''
    def __init__(self, dim: int,
                out_dim: int, 
                num_heads: int=8,
                downsample: bool = True, # not downsample at stage 1
                depth: int=2, 
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
                block: nn.Module=CAEBlock,):
        super().__init__()
        self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer) if downsample else nn.Identity()
        self.proj_cls_token = nn.Linear(dim, out_dim) if downsample else nn.Identity()
        blk_dim = out_dim if downsample else dim
        
        self.blocks = nn.Sequential(*[
            CrossCAEBlock(dim=blk_dim, 
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
            for i in range(depth)
        ])

    def forward(self, xs):
        B, N, C = xs[0].shape
        clss = [x[:, 0:1, ...] for x in xs]
        H = W = int(math.sqrt(N-1))
        xs = [x[:, 1:, ...].reshape(B, H, W, C) for x in xs]
        # downsample
        xs = [self.downsample(x) for x in xs] # share weights
        clss = [self.proj_cls_token(cls) for cls in clss]
        # blocks
        B, H, W, C = xs[0].shape
        xs = [x.reshape(B, -1, C) for x in xs]
        xs = [torch.cat((cls, x), dim=1) for cls, x in zip(clss, xs)]
        xs = self.blocks(xs)
        return xs


### -------------------------
# Model
### -------------------------
class CrossStageModel(nn.Module):
    '''
    stages of B = [2, 2, 6, 2], [2, 2, 18, 2], maybe [2, 2, 30, 2]
    num_heads = [3, 6, 12, 24] -> channels [128, 256, 512, 1024]
    each stage:
        x = downsample(x) # reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).flatten(3) > B(H//2)(W//2)(C*4)
        x = blocks(x) # first reshape to BNC, reshape(B, -1, C), process and reshape back to BHWC form
    '''
    def __init__(self, 
                img_size: int=112,
                embed_len: int=784,
                embed_dim: int=[128, 256, 512],
                depthes: List[int]=[2, 18, 2],
                num_heads: List[int]=[4, 8, 16],
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
        embed_dim = [128] + embed_dim

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[0])) for i in range(2)])
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim[0])) for i in range(2)])

        self.stages = nn.Sequential(*[CrossStageBlock(
            dim=embed_dim[i],
            out_dim=embed_dim[i+1], 
            num_heads=num_heads[i],
            downsample=i>0, # not downsample at stage 1
            depth=depthes[i], 
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
            block=block)
            for i in range(3)])
        self.norms = nn.ModuleList([norm_layer(embed_dim[-1]) for i in range(2)])
        self.heads = nn.ModuleList([SEhead(dim=embed_dim[-1], num_classes=num_classes, head_drop=head_drop) for i in range(2)])

    def _forward_features(self, x):
        # stem
        x_irs, x_lms = self.feature_extractor(x)
        x_ir, _, _ = x_irs
        x_lm, _, _ = x_lms
        xs = [x_ir, x_lm]

        # consist cls token and positional embedding
        xs = [x.flatten(2).transpose(-2, -1) for x in xs]
        x_clss = [x_cls.expand(xs[0].shape[0], -1, -1) for x_cls in self.cls_token]
        xs = [torch.cat((x_cls, x), dim=1) for x_cls, x in zip(x_clss, xs)]
        xs = [x + pos for x, pos in zip(xs, self.pos_embed)]

        xs = self.stages(xs)
        xs = [norm(x) for x, norm in zip(xs, self.norms)]
        return xs
        
    def _forward_head(self, xs):
        x_cls = [x[:, 0:1, ...] for x in xs]
        ## classification head
        out = [self.heads[i](x_cls[i]) for i in range(2)]
        out = torch.mean(torch.stack(out, dim=0), dim=0)
        return out
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self._forward_head(x)
        return x.squeeze()

### -------------------------
# Create Func
### -------------------------
def get_Cross_Stage_CAE(config):
    if config.MODEL.ARCH == 'CrossStageAC-CAE':
        return _get_CrossStageCAE(config, block=CAEBlock)

def _get_CrossStageCAE(config, block=CAEBlock):
    model = CrossStageModel(img_size=config.DATA.IMG_SIZE,
                        mlp_ratio=config.MODEL.MLP_RATIO, 
                        num_classes=config.MODEL.NUM_CLASS,
                        qkv_bias=config.MODEL.QKV_BIAS,
                        qk_norm=config.MODEL.QK_NORM,
                        init_values=config.MODEL.LAYER_SCALE,
                        attn_drop=config.MODEL.ATTN_DROP,
                        proj_drop=config.MODEL.PROJ_DROP,
                        drop_path=config.MODEL.DROP_PATH,
                        head_drop=config.MODEL.HEAD_DROP,
                        block=block)
    return model

### -------------------------
# Unit Tests
### -------------------------
class UtilTests(unittest.TestCase):
    def test_downsample(self):
        B, H, W, C = 5, 28, 28, 128
        x = torch.rand((B, H, W, C))
        downsample = PatchMerging(dim=C, out_dim=C*2, norm_layer=nn.LayerNorm)
        out = downsample(x)
        print(out.shape)
        self.assertEqual(out.shape, torch.Size([B, H//2, W//2, C*2]))

class BlockTests(unittest.TestCase):
    def test_StageBlock(self):
        B, H, W, C = 5, 28, 28, 128
        B2, H2, W2, C2 = 5, 14, 14, 256
        N, N2 = H*W, H2*W2
        x = torch.rand((B, N+1, C)) # with cls token
        # with downsample
        blk = StageBlock(dim=C, out_dim=C2, downsample=True, depth=2, block=CAEBlock)
        out = blk(x)
        self.assertEqual(out.shape, torch.Size([B, N2+1, C2]))
        print(f'Given Input shaped {x.shape},\tDownsample Shape: {out.shape}')
        # w/o downsample
        blk = StageBlock(dim=C, out_dim=C2, downsample=False, depth=2, block=CAEBlock)
        out = blk(x)
        self.assertEqual(out.shape, torch.Size([B, N+1, C]))
        print(f'w/o Downsample Shape: {out.shape}')

    def test_CrossStageBlock(self):
        B, H, W, C = 5, 28, 28, 128
        B2, H2, W2, C2 = 5, 14, 14, 256
        N, N2 = H*W, H2*W2
        xs = [torch.rand((B, N+1, C)), torch.rand((B, N+1, C))]
        print(len(xs))
        # with downsample
        blk = CrossStageBlock(dim=C, out_dim=C2, downsample=True, depth=2)
        outs = blk(xs)
        for out in outs:
            self.assertEqual(out.shape, torch.Size([B, N2+1, C2]))
            print(f'Downsample Shape: {out.shape}')
        # w/o downsample
        blk = CrossStageBlock(dim=C, out_dim=C2, downsample=False, depth=2)
        outs = blk(xs)
        for out in outs:
            self.assertEqual(out.shape, torch.Size([B, N+1, C]))
            print(f'Downsample Shape: {out.shape}')

class ModelTests(unittest.TestCase):
    def test_output(self):
        B, H, W, C = 5, 112, 112, 3
        nc = 7
        img = torch.rand((B, C, H, W))
        model = CrossStageModel(num_classes=nc) # using default settings
        out = model(img)
        print(out.shape)
        self.assertEqual(out.shape, torch.Size([B, nc]))
        
    def test_backward(self):
        B, H, W, C = 5, 112, 112, 3
        nc = 7
        model = CrossStageModel(num_classes=nc) # using default settings
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr = 1e-5)
        img = torch.rand((B, C, H, W))
        target = torch.rand((B, nc))
        # test backward
        out = model(img)
        loss = criterion(out, target)
        loss.backward()
        optim.step()
        

if __name__ == '__main__':
    unittest.main()
