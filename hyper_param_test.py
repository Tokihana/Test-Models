# inline dependencies
import argparse
import unittest
# third-party dependencies
import wandb
# local dependencies
from config.config import get_config
from data.build import build_loader
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from utils import save_checkpoint, load_checkpoint, top1_accuracy, load_finetune_weights, compute_flop_params, throughput
from model import create_model
from main import train_one_epoch, validate

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/RAF-DB_CLSFERBaseline.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./optim', type=str, help='path to log')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

class BatchSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
        
    def test_batch_size(self):
        batch_list = [2**i for i in range(7)]
        batch_res = []
        logger.info(f'testing batch sizes')
        for batchsize in batch_list:
            config.defrost()
            config.DATA.BATCH_SIZE = batchsize
            config.freeze()
            train_loader, val_loader, mix_fn = build_loader(config)
            through, step_time = throughput(self.model, train_loader, logger)
            batch_res.append([batchsize, through, step_time])
            
        logger.info(batch_res)
        
class LRTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.criterion = build_criterion(config)
        
    def test_lr(self):
        lr_list = [0.1**i for i in range(1, 8)]
        lr_res = []
        logger.info(f'testing lr')
        for lr in lr_list:
            config.defrost()
            config.TRAIN.BASE_LR = lr
            config.freeze()
            self.optimizer = build_optimizer(config, self.model)
            self.lr_scheduler = build_scheduler(config, self.optimizer, len(self.train_loader))
            max_acc = self._train_10epoch()
            lr_res.append([lr, max_acc])
        logger.info(lr_res)
        
    def _train_10epoch(self):
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            train_one_epoch(config=config, model=self.model, data_loader=self.train_loader, epoch=epoch, mix_fn = self.mix_fn, criterion=self.criterion, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, logger=logger)
            if self.val_loader is not None:
                acc, loss = validate(config, self.model, self.val_loader, logger)
                max_acc = max(max_acc, acc)
            
        return max_acc
    
class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = create_model(args, config)
        self.model.cuda()
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.criterion = build_criterion(config)
    
if __name__ == '__main__':
    # make config
    args, config = parse_option()
    # make logger
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    
    unittest.main()