# inline  
import unittest  
from typing import Optional
# third-party
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale


### ------------------------
# utils
### ------------------------
class FC(nn.Module):
    '''
    input: 
        x (B, N, C)
    '''
    def __init__(self, in_chans, out_chans, kernel_size, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
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
            
class Mlp(nn.Module):
    '''
    1d convlution mlp, input x (B, N, C)
    '''
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Conv1d(dim, int(dim*mlp_ratio), kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(int(dim*mlp_ratio), dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim)
    def forward(self, x):
        x = self.norm(x)
        x = self.bn(self.fc2(self.act(self.fc1(x.permute(0, 2, 1))))).permute(0, 2, 1)
        return x


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
        self.mlp = mlp_layer(dim=dim, mlp_ratio=mlp_ratio)
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
        else:
            # B1C/BNC + BNC
            x_gate = x_c + self.fc_encoder(x)
            
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
    

### ------------------------
# unittests
### ------------------------
class FCTests(unittest.TestCase):
    '''
    for a given FC layer with inputs shaped (B, N, C),
    suppose outputs shaped (B, N, C)
    '''
    def test_fc_layer(self):
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        fc_layer = FC(in_chans=512, out_chans=512, kernel_size=1, stride=1)
        out = fc_layer(x)
        self.assertEqual(out.shape, torch.Size([B, N, C]))
        
    def test_mlp_layer(self):
        B, N, C = 5, 49, 512
        x = torch.rand((B, N, C))
        fc_layer = Mlp(dim=512, mlp_ratio=4.)
        out = fc_layer(x)
        self.assertEqual(out.shape, torch.Size([B, N, C]))

class ClassAttentionTests(unittest.TestCase):
    '''
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

    
if __name__ == '__main__':
    unittest.main()