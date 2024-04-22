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
from model.ir50_multi import iresnet50_multi
from model.cross_cls_fusion import get_CrossCLSFER

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./optim', type=str, help='path to log')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

class MultiScaleResNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ir_back = iresnet50_multi()
        self.input = torch.rand((1, 3, 112, 112))
    def test_running(self): # note: all test def name should start with 'test'
        out = self.ir_back(self.input)
        print([o.shape for o in out])
    
class CrossCLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.CrossFER = get_CrossCLSFER(config)
        self.input = torch.rand((5, 3, 112, 112)) # batch size must larger than 1, for batch norm needs
    def test_running(self):
        out = self.CrossFER(self.input)
        print(out)

if __name__ == '__main__':
    # make config
    args, config = parse_option()
    # make logger
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    
    unittest.main()