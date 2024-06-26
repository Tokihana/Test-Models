from typing import Optional
# third-party dependencies
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
# local dependencies
from .ir50_stage3 import iresnet50_stage3

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


class CLSAttention(nn.Module):
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
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1) # dim that will compute softmax, every slice along this dim will sum to 1, for attn case, is N
        attn = self.attn_drop(attn)
        
        # BH1N @ BHN(C/H) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
               
    
class NonMultiCLSBlock(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                has_mlp: bool=True):
        super(NonMultiCLSBlock, self).__init__()
        self.has_mlp = has_mlp
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
        x_cls = x[:, 0:1, ...] + self.drop_path1(self.attn(self.norm1(x)))
        new_x = torch.cat((x_cls, x[:, 1:, ...].clone()), dim=1)
        if self.has_mlp:
            x = new_x + self.drop_path2(self.mlp(self.norm2(new_x)))
        return x
    
class NonMultiCLSBlock_onlyCLS(nn.Module):
    def __init__(self, dim: int,
                num_heads: int=8,
                mlp_ratio: float=4.,
                qkv_bias: bool=False,
                attn_drop: float=0.,
                proj_drop: float=0.,
                drop_path: float=0.,
                act_layer: nn.Module=nn.GELU,
                norm_layer: nn.Module=nn.LayerNorm,
                has_mlp: bool=True):
        super(NonMultiCLSBlock_onlyCLS, self).__init__()
        self.has_mlp = has_mlp
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
        x = x[:, 0:1, ...] + self.drop_path1(self.attn(self.norm1(x)))
        
        if self.has_mlp:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    

    
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
                block: nn.Module=NonMultiCLSBlock):
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
                 init_values: Optional[float]=None,
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 block: nn.Module=NonMultiCLSBlock):
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
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
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
                 depth: int = 4, # follows poster settings, small=4, base=6, large=8
                 num_classes: int = 7,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,):
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
                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
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
    
    
def get_NonMultiCLSFER_stage3(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                                      num_classes=config.MODEL.NUM_CLASS, 
                                      depth=config.MODEL.DEPTH, 
                                      mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,
                                      proj_drop=config.MODEL.PROJ_DROP,
                                      drop_path=config.MODEL.DROP_PATH,
                                     block=NonMultiCLSBlock)
    return model

def get_NonMultiCLSFER_onlyCLS(config):
    model = NonMultiCLSFER_stage3(img_size=config.DATA.IMG_SIZE,
                                      num_classes=config.MODEL.NUM_CLASS, 
                                      depth=config.MODEL.DEPTH, 
                                      mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,
                                      proj_drop=config.MODEL.PROJ_DROP,
                                      drop_path=config.MODEL.DROP_PATH,
                                     block=NonMultiCLSBlock_onlyCLS)
    return model


def get_MultiScaleCLSFER(config):
    model = MultiScaleCLSFER(img_size=config.DATA.IMG_SIZE,
                            num_classes=config.MODEL.NUM_CLASS, 
                            depth=config.MODEL.DEPTH, 
                            mlp_ratio=config.MODEL.MLP_RATIO,
                            attn_drop=config.MODEL.ATTN_DROP,
                            proj_drop=config.MODEL.PROJ_DROP,
                            drop_path=config.MODEL.DROP_PATH,)
    return model

