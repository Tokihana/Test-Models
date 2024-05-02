# inline dependencies
import argparse
import unittest
# third-party dependencies
import torch
import torch.nn as nn
# local dependencies
from model.cls_vit import CLSAttention, CLSBlock
from model.ir50 import iresnet50
from model.ir50_poster import Backbone
from model.baseline import Baseline
from data.build import build_loader
from config.config import get_config
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from utils import save_checkpoint, load_checkpoint, top1_accuracy, load_finetune_weights, compute_flop_params, throughput
from model import create_model
from main import validate

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./optim', type=str, help='path to log')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

class DataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config.defrost()
        config.DATA.BATCH_SIZE=5
        config.freeze()
        self.train_loader, self.val_loader, _ = build_loader(config)
    def test_outputs(self):
        for images, targets in self.train_loader:
            break
        print(images, targets)

class MixupTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        
    def test_mix_out(self):
        for images, targets in self.train_loader:
            logger.info(f'Before mix: {images.shape}, {targets.shape}')
            images, targets = self.mix_fn(images, targets)
            logger.info(f'After mix: {images.shape}, {targets.shape}')
            logger.info(targets)
            break
            
class ConfusionMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.model = create_model(args, config).cuda()
        
    def test_confusion_matrix(self):
        config.defrost()
        config.TRAIN.CONFUSION_MATRIX = True
        config.freeze()
        validate(config, self.model, self.val_loader, logger)
        

class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
    def test_load_backbone(self):
        irback = iresnet50(num_features=7)
        checkpoint = torch.load('./model/pretrain/ir50_backbone.pth')
        miss, unexcepted = irback.load_state_dict(checkpoint, strict=False)
        logger.info(f'Miss: {miss},\t Unexcept: {unexcepted}')
        del checkpoint
        
class NonMultiCLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
        self.input = torch.rand((1, 3, 224, 224)).cuda()
        #self.input = torch.rand((1, 3, 224, 224))
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        
    def test_model(self):
        out = self.model(self.input) 
        logger.info(out.shape)
    
    def test_flops(self):
        throughput(self.model, self.train_loader, logger)
        
class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.input = torch.rand((1, 3, 112, 112)).cuda()
    
    def test_running(self):
        out = self.model(self.input)
        for images, targets in self.train_loader:
            images = images.cuda()
            out = self.model(images)
            print(f'out shape: {out.shape}')
            break

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
        
class MultiScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config.defrost()
        config.MODEL.ARCH = 'MultiScaleCLSFER'
        config.freeze()
        self.model = create_model(args, config)
        self.input = torch.rand((1, 3, 224, 224))
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
    def single_fig_running_test(self):
        logger.info(self.model.parameters())
        output = self.model(self.input)
        logger.info(output)
        logger.info(output.shape)
    def loader_running_test(self):
        for images, targets in self.val_loader:
            break
        logger.info(images.shape)
        logger.info(targets.shape)
        output = self.model(images)
        logger.info(output.shape)
        
class CLSFER_14x14_Tests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.criterion = nn.CrossEntropyLoss()
    def test_14x14_models(self):
        test_input = torch.rand((5, 3, 112, 112))
        test_target = torch.rand((5, config.MODEL.NUM_CLASS))
        for arch in ['14x14_CLSFER_baseline', '14x14_CLSFER_catAfterMlp', '14x14_CLSFER_addpatches']:
            config.defrost()
            config.MODEL.ARCH = arch
            config.freeze()
            model = create_model(args, config)
            out = model(test_input)
            loss = self.criterion(out, test_target)
            logger.info(f'Arch: {arch}, out: {out.shape}, loss: {loss:.3f}')
    def test_14x14se_models(self):
        test_input = torch.rand((5, 3, 112, 112))
        test_target = torch.rand((5, config.MODEL.NUM_CLASS))
        for arch in ['14x14se_CLSFER_baseline', '14x14se_CLSFER_catAfterMlp', '14x14se_CLSFER_addpatches']:
            config.defrost()
            config.MODEL.ARCH = arch
            config.freeze()
            model = create_model(args, config)
            out = model(test_input)
            loss = self.criterion(out, test_target)
            logger.info(f'Arch: {arch}, out: {out.shape}, loss: {loss:.3f}')
            
class CLSFER_28x28_Tests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.criterion = nn.CrossEntropyLoss()
    def test_28x28_models(self):
        test_input = torch.rand((5, 3, 112, 112))
        test_target = torch.rand((5, config.MODEL.NUM_CLASS))
        for arch in ['28x28_CLSFER_baseline', '28x28_CLSFER_catAfterMlp', '28x28_CLSFER_addpatches']:
            config.defrost()
            config.MODEL.ARCH = arch
            config.freeze()
            model = create_model(args, config)
            out = model(test_input)
            loss = self.criterion(out, test_target)
            logger.info(f'Arch: {arch}, out: {out.shape}, loss: {loss:.3f}')
    def test_28x28se_models(self):
        test_input = torch.rand((5, 3, 112, 112))
        test_target = torch.rand((5, config.MODEL.NUM_CLASS))
        for arch in ['28x28se_CLSFER_baseline', '28x28se_CLSFER_catAfterMlp', '28x28se_CLSFER_addpatches']:
            config.defrost()
            config.MODEL.ARCH = arch
            config.freeze()
            model = create_model(args, config)
            out = model(test_input)
            loss = self.criterion(out, test_target)
            logger.info(f'Arch: {arch}, out: {out.shape}, loss: {loss:.3f}')

if __name__ == '__main__':
    # make config
    args, config = parse_option()
    # make logger
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    
    unittest.main()