# inline dependencies
import unittest
# third-party dependencies
import torch
# local dependencies
from model.cls_vit import CLSAttention, CLSBlock
from model.ir50 import iresnet50

class IrBackTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        B, C, H, W = 5, 3, 224, 224
        self.input = torch.rand((B, C, H, W))
        self.back = iresnet50(num_features=7)
    
    def test_outputs(self):
        xs = self.back(self.input)
        for x in xs:
            print(x.shape)
    

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