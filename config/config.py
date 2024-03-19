# config.py
import os
from yacs.config import CfgNode as CN

# root node
_C = CN()

## ----------------------------------------------
# SYSTEM settings
## ----------------------------------------------
_C.SYSTEM = CN()
# emperimental information
_C.SYSTEM.EXPERIMENT_PATH = './results/minitest'
# log path, joint with experiment path
_C.SYSTEM.LOG = 'log'
# checkpoint path, joint with experiment path
_C.SYSTEM.CHECKPOINT = 'checkpoint'
# save frequency
_C.SYSTEM.SAVE_FREQ = 20
# print frequency
_C.SYSTEM.PRINT_FREQ = 30

## ----------------------------------------------
# DATA settings
## ----------------------------------------------
_C.DATA = CN()
# datasets configs
# name of dataset, RAF-DB for default, supported: RAF-DB, AffectNet_7, AffectNet_8
_C.DATA.DATASET = 'RAF-DB'
# path to dataset
_C.DATA.DATA_PATH = '../datasets/RAF-DB'
# loader configs
# image size
_C.DATA.IMG_SIZE = 224
# batch size
_C.DATA.BATCH_SIZE = 64
# num of workers
_C.DATA.NUM_WORKERS = 8
# use pin memory or not
_C.DATA.PIN_MEMORY = True

## ----------------------------------------------
# MODEL settings
## ----------------------------------------------
_C.MODEL = CN()
# model name, support IResNet50, RepVGGs
_C.MODEL.ARCH = 'IResNet50'
# default nb_classes
_C.MODEL.NUM_CLASS = 7


## ----------------------------------------------
# MODE settings
## ----------------------------------------------
_C.MODE = CN()
# EVAL flas
_C.MODE.EVAL = False
# FINETUEN flags
_C.MODE.FINETUNE = False
# THROUGHPUT flags
_C.MODE.THROUGHPUT = False
# FLOP flags
_C.MODE.FLOP = False

## ----------------------------------------------
# TRAINING settings
## ----------------------------------------------
_C.TRAIN = CN()
# epochs
_C.TRAIN.EPOCHS = 200
# start epoch
_C.TRAIN.START_EPOCH = 0
# warmup epochs
_C.TRAIN.WARMUP_EPOCHS = 20
# weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# base lr
_C.TRAIN.BASE_LR = 3.5e-5
# warmup lr
_C.TRAIN.WARMUP_LR = 0.0
# min lr
_C.TRAIN.MIN_LR = 0
# resume checkpoint path
_C.TRAIN.RESUME = ''

# criterion
_C.TRAIN.CRITERION = CN()
# type of criterion, support CrossEntropy, LabelSmoothing, SoftTargetCE
_C.TRAIN.CRITERION.NAME = 'CrossEntropy'
# Label Smoothing
_C.TRAIN.CRITERION.LABEL_SMOOTHING = 0.1

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'exponential'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma for Expoential Scheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.98

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adam'
# epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

## ----------------------------------------------
# AUGMENTATION settings
## ----------------------------------------------
_C.AUG = CN()
_C.AUG.MIXUP = 0.2

## ----------------------------------------------
# TEST settings
## ----------------------------------------------

def get_config_default():
    config = _C.clone()
    return config

def get_config(args):
    config = get_config_default()
    config.defrost()
    config.merge_from_file(args.config)

    # build experiment folder
    if not config.SYSTEM.EXPERIMENT_PATH == '':
        if not os.path.exists(config.SYSTEM.EXPERIMENT_PATH):
            os.makedirs(config.SYSTEM.EXPERIMENT_PATH)
        config.SYSTEM.LOG = os.path.join(config.SYSTEM.EXPERIMENT_PATH, config.SYSTEM.LOG)
        config.SYSTEM.CHECKPOINT = os.path.join(config.SYSTEM.EXPERIMENT_PATH, config.SYSTEM.CHECKPOINT)
        if not os.path.exists(config.SYSTEM.CHECKPOINT):
            os.makedirs(config.SYSTEM.CHECKPOINT) 
        if not os.path.exists(config.SYSTEM.LOG):
            os.makedirs(config.SYSTEM.LOG)
    
    config.freeze()
    
    return config
    
    return config