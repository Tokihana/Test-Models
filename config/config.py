# config.py
import os
import datetime
from yacs.config import CfgNode as CN

# root node
_C = CN()

## ----------------------------------------------
# SYSTEM settings
## ----------------------------------------------
_C.SYSTEM = CN()
# project name, for w&b
_C.SYSTEM.PROJECT_NAME = ''
# experiment name
_C.SYSTEM.EXPERIMENT_NAME = ''
# emperiment path
_C.SYSTEM.EXPERIMENT_PATH = './results/minitest'
# log path, joint with experiment path
_C.SYSTEM.LOG = 'log'
# checkpoint path, joint with experiment path
_C.SYSTEM.CHECKPOINT = 'checkpoint'
# save frequency
_C.SYSTEM.SAVE_FREQ = 50
# print frequency
_C.SYSTEM.PRINT_FREQ = 100

## ----------------------------------------------
# DATA settings
## ----------------------------------------------
_C.DATA = CN()
# datasets configs
# name of dataset, RAF-DB for default, supported: RAF-DB, AffectNet_7, AffectNet_8, FERPlus, JAFFE, CK+
_C.DATA.DATASET = 'RAF-DB'
# path to dataset
_C.DATA.DATA_PATH = '../datasets/RAF-DB'
# erasing scale
_C.DATA.ERASING_SCALE = (0.02, 0.1)
# erasing p
_C.DATA.ERASING_P = 0.5
# loader configs
# image size
_C.DATA.IMG_SIZE = 224
# batch size
_C.DATA.BATCH_SIZE = 64
# val batch size
_C.DATA.VAL_BATCH_SIZE = 200
# num of workers
_C.DATA.NUM_WORKERS = 8
# use pin memory or not
_C.DATA.PIN_MEMORY = True
# use Imbalance sampler or not
_C.DATA.IMBALANCED_SAMPLER = False

## ----------------------------------------------
# MODEL settings
## ----------------------------------------------
_C.MODEL = CN()
# model name, support IResNet50, RepVGGs, ViT-FER baseline, NonMultiCLSFER
_C.MODEL.ARCH = 'IResNet50'
# default nb_classes
_C.MODEL.NUM_CLASS = 7
# encoder depth
_C.MODEL.DEPTH = 8
# mlp ratio
_C.MODEL.MLP_RATIO = 4.
# attn drop
_C.MODEL.ATTN_DROP = 0.
# proj drop
_C.MODEL.PROJ_DROP = 0.
# drop path
_C.MODEL.DROP_PATH = 0.
# head drop
_C.MODEL.HEAD_DROP = 0.
# qk norm
_C.MODEL.QK_NORM = True
## Layer Scale
_C.MODEL.LAYER_SCALE = 1e-5
# token SE
_C.MODEL.TOKEN_SE = 'linear'
# qkv bias
_C.MODEL.QKV_BIAS = False

# mlp layer, 'linear' or 'conv1d'
_C.MODEL.MLP_LAYER = 'linear'
# fc layer, 'linear' or 'conv1d'
_C.MODEL.FC_LAYER = 'linear'

# StarBlock ablations
_C.MODEL.STARBLOCK = CN()
## gate, can be 'CAE' or 'FC'
_C.MODEL.STARBLOCK.GATE = 'CAE'
## use 'star' or 'sum'
_C.MODEL.STARBLOCK.USE_STAR = True
## skip or dense short cut connection  
_C.MODEL.STARBLOCK.SHORTCUT = 'dense'



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
# tSNE 
_C.MODE.T_SNE = False

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
_C.TRAIN.MIN_LR = 0.
# resume checkpoint path
_C.TRAIN.RESUME = ''
# whether to use confusion matrix
_C.TRAIN.CONFUSION_MATRIX = False

# criterion
_C.TRAIN.CRITERION = CN()
## type of criterion, support CrossEntropy, LabelSmoothing, SoftTargetCE
_C.TRAIN.CRITERION.NAME = 'CrossEntropy'
## Label Smoothing
_C.TRAIN.CRITERION.LABEL_SMOOTHING = 0.1
## focal gamma
_C.TRAIN.CRITERION.FOCAL_GAMMA = 2.

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'exponential'
## Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
## LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
## Gamma for Expoential Scheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.98
## factor for Reduce On Plateau
_C.TRAIN.LR_SCHEDULER.REDUCE_FACTOR = 0.5

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adam'
## epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
## betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
## SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
## SAM rho
_C.TRAIN.OPTIMIZER.RHO = 0.05

## ----------------------------------------------
# AUGMENTATION settings
## ----------------------------------------------
_C.AUG = CN()
# mixup alpha, default 1.0
_C.AUG.MIXUP = 1.0

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
        now = datetime.datetime.now()
        time_str = now.strftime("[%m-%d]-[%H-%M]")
        config.SYSTEM.EXPERIMENT_PATH = f'{config.SYSTEM.EXPERIMENT_PATH}_{time_str}'
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