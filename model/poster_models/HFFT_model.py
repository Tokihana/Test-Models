from ast import Num
from ctypes.wintypes import SHORT
import torch
import torch.nn as nn
from .window_blocks import WindowAttentionGlobal, window
from .backbones import Feature_Extractor
from .raf_db_loader import RAF_DB_Loader
from .posterv2 import FFN, _to_channel_first, _to_query
from .vit_model import PatchEmbed, VisionTransformer

def lm_to_q(lm):
    B, C, H, W = lm.shape
    return lm.permute(0, 2, 3, 1).reshape(B, H*W, C)

class HFFT(nn.Module):
    def __init__(self, window_size=[28, 14, 7], dims=[64, 128, 256],
                 num_heads = [2, 4, 8]):
        super().__init__()
        self.window_sizes = window_size
        self.dims = dims
        self.N = [w * w for w in window_size]
        self.dim_heads = [int(torch.div(dim, num_head).item()) for dim, num_head in zip(dims, num_heads)]  
        self.num_heads = num_heads
        self.nfs = len(dims) # num of feature scales

        self.fea_ext = Feature_Extractor()
        self.windows = nn.ModuleList([window(self.window_sizes[i], dims[i]) for i in range(self.nfs)])
        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]
        self.attn_GCIP = nn.ModuleList([WindowAttentionGlobal(dims[i], num_heads[i],window_size[i]) for i in range(self.nfs)])
        self.ffn_GCIP = nn.ModuleList([FFN(dims[i],window_size[i],layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.nfs)])
        self.attn_GCIA = nn.ModuleList([WindowAttentionGlobal(dims[i], num_heads[i],window_size[i]) for i in range(self.nfs)])
        self.ffn_GCIA = nn.ModuleList([FFN(dims[i],window_size[i],layer_scale=1e-5, drop_path=dpr[i]) for i in range(self.nfs)])
        # ViT
        self.embed_q = nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        self.embed_k = nn.Sequential(nn.Conv2d(dims[1], 768, kernel_size=3, stride=2, padding=1))
        self.embed_v = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)
        self.VIT = VisionTransformer(depth=2, embed_dim=768)
        
    def forward(self, x):
        lms, irs = self.fea_ext(x)
        # GCIP
        # split irs to windows, then use the original shortcut as query
        window_splits = [list(self.windows[i](irs[i])) for i in range(self.nfs)]
        x_windows, shortcuts = [c[0] for c in window_splits], [c[1] for c in window_splits]
        q_irgs = [_to_query(shortcuts[i], self.N[i], self.num_heads[i], self.dim_heads[i]) for i in range(self.nfs)] # irs global query
        gcip_outs = [self.attn_GCIP[i](x_windows[i], q_irgs[i]) for i in range(self.nfs)]
        ffn_gcip_outs = [self.ffn_GCIP[i](gcip_outs[i], shortcuts[i]) for i in range(self.nfs)]
        
        # FSFP
        # use gcip_outs as query, lms as kv
        q_gcips = [_to_query(gcip_outs[i], self.N[i], self.num_heads[i], self.dim_heads[i]) for i in range(self.nfs)]
        q_lms = [lm_to_q(lm) for lm in lms]
        gcia_outs = [self.attn_GCIA[i](q_lms[i], q_gcips[i]) for i in range(self.nfs)]
        ffn_gcia_outs = [self.ffn_GCIA[i](gcia_outs[i], lms[i].permute(0, 2, 3, 1)) for i in range(self.nfs)]
        
        outs = [_to_channel_first(o) for o in ffn_gcip_outs]
        o1, o2, o3 = self.embed_q(outs[0]).flatten(2).transpose(1, 2), self.embed_k(outs[1]).flatten(2).transpose(1, 2), self.embed_v(outs[2])
        o = torch.cat([o1, o2, o3], dim=1)
        out = self.VIT(o)
        return out
    
        
        
        
        
        
        