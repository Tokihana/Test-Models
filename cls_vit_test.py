# inline dependencies
import unittest
# third-party dependencies
import torch
import torch.nn as nn
# local dependencies
from model.cls_vit import CLSAttention, CLSBlock
from model.ir50 import iresnet50
from model.ir50_poster import Backbone
from model.baseline import Baseline

class IrBackTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        B, C, H, W = 5, 3, 224, 224
        self.input = torch.rand((B, C, H, W))
        self.back = iresnet50(num_features=7)
        self.back_poster = Backbone(50, 0, 'ir')
    
    def test_outputs(self):
        xs = self.back(self.input)
        print(type(xs))
        print(xs.shape)
        x_poster = self.back_poster(self.input)
        print(x_poster[0].shape, x_poster[1].shape)
    
    def test_conv1d(self):
        x1 = torch.rand((5, 49, 1024))
        x2 = torch.rand((5, 49, 1024))
        x1 = torch.cat((x1, torch.mean(x1, 1).view(5,1,-1)), dim=1)
        x2 = torch.cat((x2, torch.mean(x2, 1).view(5,1,-1)), dim=1)
        print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        print(x.shape)
        
        n_channels = x.shape[1]
        downsample_m = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)
        downsample_s = nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=4)
        print(downsample_m(x).shape, downsample_s(x).shape)
        
    def test_view(self):
        x = torch.range(1, 36).reshape(1, 6, 6)
        #x = torch.tensor([[[1, 2], [3, 4],]])
        print('test view:',)
        print(x)
        print('after view:')
        print(x.view(9, 4))
        
class HybridViTBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        B, C, H, W = 5, 3, 224, 224
        self.input = torch.rand((B, C, H, W))
        self.baseline = Baseline()
    
    def test_framwork(self):
        output = self.baseline(self.input)
        print(output.shape)
        

class CLSViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        B, N, C = 3, 50, 128
        self.input = torch.rand((B, N, C))
        print(f'Input shape: {self.input.shape}')
        
    def test_attn(self):
        '''
        test Attention object
        '''
        B, N, C = self.input.shape
        attn = CLSAttention(C)
        out = attn(self.input)
        print(f'After Attn shape: {out.shape}')
        self.assertEqual(out.shape, torch.Size([B, 1, C]))
    
    def test_CLSblock(self):
        B, N, C = self.input.shape
        block = CLSBlock(C, num_heads=8)
        out = block(self.input)
        print(f'After Block shape: {out.shape}')
        self.assertEqual(out.shape, torch.Size([B, 1, C]))
        
            

if __name__ == '__main__':
    unittest.main()