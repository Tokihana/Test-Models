# inline  
import unittest  
from typing import Optional
# third-party
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale
# local
from .ir50 import iresnet50

__all__ = ['StarBlock', 'SingleModel', 'get_stars']

### ------------------------
# utils
### ------------------------
class FC(nn.Module):
    '''
    input: 
        x (B, N, C)
    '''
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, with_bn: bool=True):
        super().__init__()
        self.conv = nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if with_bn:
            self.bn = nn.BatchNorm1d(out_chans)
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0) 
        else:
            self.bn = nn.Identity()
    def forward(self, x):
        x = self.bn(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        return x
    
class SE_block(nn.Module):
    def __init__(self, channels, rd_ratio=1./16, rd_divisor=8, bias=True):
        super().__init__()
        from timm.layers.helpers import make_divisible
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
            
class Mlp(nn.Module):
    '''
    1d convlution mlp, input x (B, N, C)
    '''
    def __init__(self, dim: int, 
                 mlp_ratio: float=4.,
                 drop: float=0.,
                 act_layer: nn.Module=nn.GELU,):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Conv1d(dim, int(dim*mlp_ratio), kernel_size=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Conv1d(int(dim*mlp_ratio), dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.norm(x)
        x = self.drop1(self.act(self.fc1(x.permute(0, 2, 1))))
        x = self.drop2(self.bn(self.fc2(x))).permute(0, 2, 1)
        return x
    
class SEhead(nn.Sequential):
    def __init__(self, dim: int, num_classes: int, head_norm: bool=True, head_se: bool=True, head_drop: float=0.):
        super().__init__()
        if head_norm:
            self.add_module('head_norm', nn.LayerNorm(dim))
        if head_se:
            self.add_module('se_block', SE_block(channels=dim))
        if head_drop > 0.:
            self.add_module('head_drop', nn.Dropout(head_drop))
        self.add_module('linear', nn.Linear(dim, num_classes))


### ------------------------
# Class Attention
### ------------------------
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
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = attn.softmax(dim=-1) # dim that will compute softmax, every slice along this dim will sum to 1, for attn case, is N
        attn = self.attn_drop(attn)
        
        # BH1N @ BHN(C/H) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x   
    
    

### ------------------------
# Star Block
### ------------------------
class StarBlock(nn.Module):
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
                mlp_layer: nn.Module = Mlp,
                gate: str='CAE', # 'CAE' or FC
                use_star: bool=True, 
                shortcut: str='skip', # 'skip' or 'dense'
                ):
        super().__init__()
        self.gate = gate
        self.use_star = use_star
        self.shortcut = shortcut
        # gate branch
        self.norm1 = norm_layer(dim)
        if self.gate == 'CAE':
            self.gate_layer = CLSAttention(dim, 
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                qk_norm=qk_norm,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
                                norm_layer=norm_layer,)
        else: # self.gate == 'FC'
            self.gate_layer = nn.Sequential(FC(in_chans=dim, out_chans=dim, kernel_size=1, stride=1),
                                            act_layer())
        # fc branch
        self.fc_encoder = FC(in_chans=dim, out_chans=dim, kernel_size=1, stride=1)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # mlp after gate
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        '''
        x (B, N, C)
        '''
        inputs = x
        # BNC -> B1C/BNC
        x_c = self.drop_path1(self.ls1(self.gate_layer(self.norm1(x))))
        
        # star after encoder
        if self.use_star:
            # B1C/BNC * BNC
            x_gate = x_c * self.fc_encoder(x)
        else: # applied sum to update the patch tokens 
            # B1C/BNC + BNC
            x_gate = x_c + self.fc_encoder(x)
        
        if self.shortcut == 'dense':
            x_gate = x_gate + inputs
            
        # mlp
        x_mlp = self.drop_path2(self.ls2(self.mlp(self.norm2(x_gate))))
        
        # shortcut
        if self.shortcut == 'skip':
            x = inputs + x_mlp
        else: # self.shortcut == 'dense'
            x = inputs + x_gate + x_mlp
            
        return x
    
    
### ------------------------
# models
### ------------------------
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
                block: nn.Module=StarBlock,
                **kwargs):
        super().__init__()
        # ir50 backbone
        self.irback = iresnet50(num_features=num_classes)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = self.irback.load_state_dict(checkpoint, strict=False)
        print(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint, miss, unexcepted
        
        # running blocks
        ## cls token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1, embed_dim))
        ## stochastic depth drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        ## blocks 
        if block == StarBlock: # using star block
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
                      gate=kwargs['gate'],
                      use_star=kwargs['use_star'],
                      shortcut=kwargs['shortcut'])
            for i in range(depth)])
        else:
            raise ModuleNotFoundError(f'unsupported block type {block}')
        
        self.norm = norm_layer(embed_dim)
        self.head = SEhead(dim=embed_dim, num_classes=num_classes, head_drop=head_drop)
        
    def forward(self, x):
        # get stem output
        _, x_ir, _ = self.irback(x)
        x_embed = x_ir.flatten(2).transpose(-2, -1)
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



        
### ------------------------
# model create  
### ------------------------
def get_stars(config):
    if config.MODEL.ARCH == 'starCAE-single':
        return _get_single_model(config)

def _get_single_model(config):
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
                        block=StarBlock,
                        gate=config.MODEL.STARBLOCK.GATE,
                        use_star=config.MODEL.STARBLOCK.USE_STAR,
                        shortcut=config.MODEL.STARBLOCK.SHORTCUT)
    return model
   
                        
### ------------------------
# unittests
### ------------------------
class UtilsTests(unittest.TestCase):
    '''
    Tests for util classes.
    '''
    def test_fc_layer(self):
        '''
        for a given FC layer with inputs shaped (B, N, C),
        suppose outputs shaped (B, N, C)
        '''
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        fc_layer = FC(in_chans=512, out_chans=512, kernel_size=1, stride=1)
        out = fc_layer(x)
        self.assertEqual(out.shape, torch.Size([B, N, C]))
        
    def test_mlp_layer(self):
        '''
        given inputs shaped (B, N, C),
        suppose outputs shaped (B, N, C)
        '''
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        mlp_layer = Mlp(dim=512, mlp_ratio=4.)
        out = mlp_layer(x)
        self.assertEqual(out.shape, torch.Size([B, N, C]))
    
    def test_head(self):
        '''
        given features shaped (B, 1, C),
        suppose outputs shaped (B, num_class)
        '''
        B, C, num_class = 5, 512, 7 
        x = torch.rand((B, 1, C)) # class token
        head = SEhead(C, num_class)
        self.assertEqual(head(x).squeeze().shape, torch.Size([B, num_class]))

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
        
class StarBlockTests(unittest.TestCase):
    '''
    test Star Block ablations.
    given inputs shaped (B, N, C),
    suppose outputs shaped (B, N, C)
    '''
    @classmethod
    def setUpClass(self):
        self.B, self.N, self.C = 5, 49, 512
        self.x = torch.rand((self.B, self.N, self.C))
    
    def test_fc_sum_skip(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='FC', use_star=False, shortcut='skip')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_fc_sum_dense(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='FC', use_star=False, shortcut='dense')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_fc_star_skip(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='FC', use_star=True, shortcut='skip')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_fc_star_dense(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='FC', use_star=True, shortcut='dense')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_CAE_sum_skip(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='CAE', use_star=False, shortcut='skip')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_CAE_sum_dense(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='CAE', use_star=False, shortcut='dense')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_CAE_star_skip(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='CAE', use_star=True, shortcut='skip')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
    def test_CAE_star_dense(self):
        # fc gate, sum, skip shortcut
        block = StarBlock(dim=self.C, gate='CAE', use_star=True, shortcut='dense')
        self.assertEqual(block(self.x).shape, torch.Size([self.B, self.N, self.C]))
        
class ModelTests(unittest.TestCase):
    '''
    test models, given inputs shaped (B, C, H, W), 
    suppose outputs shaped (B, num_class
    '''
    def test_single_model(self):
        B, C, H, W = 5, 3, 112, 112
        num_class = 7
        images = torch.rand((B, C, H, W))
        model = SingleModel(num_classes=num_class, gate='CAE', use_star=True, shortcut='dense')
        out = model(images)
        self.assertEqual(out.shape, torch.Size([B, num_class]))
    
if __name__ == '__main__':
    '''
    unittest may raise error at:
    1. [line 9]: from .ir50 import iresnet50
       just replace '.ir50' as 'ir50'
    2. [line 250]: checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
       relative path, delete '/model'
    '''
    unittest.main()