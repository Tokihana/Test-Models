import argparse
import unittest

from config.config import get_config
from data.build import build_loader
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from utils import save_checkpoint, load_checkpoint, top1_accuracy, load_finetune_weights, compute_flop_params, throughput
from model import create_model

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
        self.model = model = create_model(args, config)
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

if __name__ == '__main__':
    # make config
    args, config = parse_option()
    # make logger
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    
    unittest.main()