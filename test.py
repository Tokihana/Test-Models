# test.py
# inline dependencies
import os
import unittest
import argparse
# third-party dependencies
import torch
import torch.nn as nn
from torchvision.transforms import v2
from timm.utils import accuracy, AverageMeter
# local dependencies
from data.build import build_loader, build_dataset, _get_rafdb_transform
from config.config import get_config
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from model import create_model
from utils import save_checkpoint, load_checkpoint

from main import validate, train_one_epoch

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/repvgg.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

class ArgConfigTests(unittest.TestCase):
    def test_args(self):
        logger.info(args)
        # check args
        self.assertTrue(args.config == './config/yaml/test.yaml')
        self.assertTrue(args.log == './log')
        
    def test_configs(self):
        logger.info(config)
        # check configs
        self.assertEqual(config.SYSTEM.LOG, './log')
        self.assertFalse(config.SYSTEM.CHECKPOINT == './checkpoint')
        self.assertEqual(config.SYSTEM.CHECKPOINT, './log')

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_fig = torch.ones((3, 14, 14))            
    
    def test_transform(self):
        logger.info(['fig size before transform:', self.test_fig.shape])
        #mean=[0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        transform, _ = _get_rafdb_transform()
        logger.info(['after transform:', transform(self.test_fig).shape])
        self.assertEqual(transform(self.test_fig).shape, torch.Size([3, 224, 224]))

    def test_dataset(self):
        train_dataset, val_dataset, num_classes = build_dataset(config=config)
        if config.DATA.DATASET == 'RAF-DB':
            self.assertEqual(train_dataset.__len__(), 12271)
            self.assertEqual(val_dataset.__len__(), 3068)
        elif config.DATA.DATASET == 'AffectNet_7':
            self.assertEqual(train_dataset.__len__(), 283901)
            self.assertEqual(val_dataset.__len__(), 3500)
        elif config.DATA.DATASET == 'AffectNet_8':
            self.assertEqual(train_dataset.__len__(), 287651)
            self.assertEqual(val_dataset.__len__(), 4000)
        elif config.DATA.DATASET == 'FERPlus':
            self.assertEqual(train_dataset.__len__(), 28386)
            self.assertEqual(val_dataset.__len__(), 3553)
        else:
            logger.info(f'DATASET {config.DATA.DATASET} NOT SUPPORTED')

    def test_dataloader_and_mixup(self):
        train_loader, val_loader, mix_fn = build_loader(config)
        for samples, targets in train_loader:
            break
        self.assertEqual(samples.shape, torch.Size([config.DATA.BATCH_SIZE, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]))
        logger.info(f'before mixup: {[samples.shape, targets.shape]}')
        samples, targets = mix_fn(samples, targets)
        logger.info(f'after mixup: {[samples.shape, targets.shape]}')

class UtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = MLP(4, 10, 3) # in_c = 4, out_c = 3
    def test_acc_avgmeter(self):
        target = torch.tensor([0, 1, 2, 3, 4]) # 5 examples, 5 class
        out1 = torch.tensor([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]])
        out2 = torch.zeros((5, 5)) 
        logger.info(accuracy(out1, target, topk=(1,)))
        logger.info(accuracy(out2, target, topk=(1,)))
        acc2 = accuracy(out2, target, topk=(1,))
        self.assertEqual(acc2, [torch.tensor(20.)])
        
        accs = AverageMeter()
        accs.update(acc2[0], target.size(0)) # .update(val, n)
        accs.update(acc2[0], target.size(0))
        self.assertEqual(accs.avg, torch.tensor(20.))
        
    def test_model_save(self):
        inputs = torch.ones((5, 4))
        logger.info(self.model(inputs).shape)
        
        states = {'state_dict': self.model.state_dict()}
        torch.save(states, config.SYSTEM.CHECKPOINT + '.pth')
        self.assertTrue(os.path.exists(config.SYSTEM.CHECKPOINT + '.pth'))
        
    def test_model_load(self):
        checkpoint = torch.load(config.MODEL.RESUME)
        self.model.load_state_dict(checkpoint['state_dict'])
        logger.info(self.model)
        
        
    # def test_flops(self):
        # not emergency for now
        
    
    def test_num_params(self):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
    
    # def test_throughput(self):

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.model = create_model(args, config)
        self.checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        # logger.info(checkpoint.keys())
        if not config.MODE.FINETUNE:
            self.model.load_state_dict(self.checkpoint)
        self.model.cuda()
        self.optimizer = build_optimizer(config, self.model)
        self.scheduler = build_scheduler(config, self.optimizer, len(self.train_loader))
        self.criterion = build_criterion(config)
        
    def test_model(self):
        logger.info('MODEL INFO:')
        logger.info(self.model)
        logger.info(self.model.parameters())
        
    def test_optimizer(self):
        logger.info('OPTIMIZER INFO:')
        logger.info(self.optimizer)
        
    def test_scheduler(self):
        logger.info('SCHEDULER INFO:')
        logger.info(self.scheduler)
    
    def test_criterion(self):
        logger.info('CRITERION INFO:')
        logger.info(self.criterion)
        
    def test_load_non_strict_checks(self):
        new_dict = self.checkpoint.copy()
        pop_list = []
        for key in new_dict.keys():
            if 'linear' in key:
                pop_list.append(key)
        logger.info(pop_list)
        for key in pop_list:
            new_dict.pop(key) # can not pop during dict iteration
            
    def test_train_one_epoch(self):
        train_one_epoch(config=config, model=self.model, data_loader=self.train_loader, epoch=0, mix_fn = self.mix_fn, criterion=self.criterion, optimizer=self.optimizer, lr_scheduler=self.scheduler, logger=logger)
        acc, loss = validate(config, self.model, self.val_loader, logger)
        
    def test_validate(self):
        self.model.cuda()
        #acc, loss = validate(config, self.model, self.val_loader, logger)
        

if __name__ == '__main__':
    # make logger
    logger = create_logger('log', name='testlog.log')
    # make config
    args, config = parse_option()
    
    unittest.main()