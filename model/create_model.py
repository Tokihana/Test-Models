import torch
from .repvggplus import create_RepVGGplus_by_name
from .iresnet import iresnet50
from .baseline import Baseline

def create_model(args, config):
    model = None
    if 'RepVGG' in config.MODEL.ARCH:
        model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint, config=config)
    elif config.MODEL.ARCH == 'IResNet50':
        model = iresnet50(num_features=config.MODEL.NUM_CLASS)
    elif config.MODEL.ARCH == 'CLSFERBaseline':
        model = Baseline(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH)
        
    return model

