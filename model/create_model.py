import torch
from .repvggplus import create_RepVGGplus_by_name
from .iresnet import iresnet50
from .baseline import Baseline
from .cls_vit import NonMultiCLSFER
from .cls_vit_stage3 import NonMultiCLSFER_stage3

def create_model(args, config):
    model = None
    if 'RepVGG' in config.MODEL.ARCH:
        model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint, config=config)
    elif config.MODEL.ARCH == 'IResNet50':
        model = iresnet50(num_features=config.MODEL.NUM_CLASS)
    elif config.MODEL.ARCH == 'CLSFERBaseline':
        model = Baseline(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH, mlp_ratio=config.MODEL.MLP_RATIO,
                        attn_drop=config.MODEL.ATTN_DROP, )
    elif config.MODEL.ARCH == 'NonMultiCLSFER':
        model = NonMultiCLSFER(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH, mlp_ratio=config.MODEL.MLP_RATIO,
                               attn_drop=config.MODEL.ATTN_DROP,)
    elif config.MODEL.ARCH == 'NonMultiCLSFER_stage3':
        model = NonMultiCLSFER_stage3(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH, mlp_ratio=config.MODEL.MLP_RATIO,
                                      attn_drop=config.MODEL.ATTN_DROP,)
        
    return model

