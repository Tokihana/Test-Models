import torch
from .repvggplus import create_RepVGGplus_by_name
from .iresnet import iresnet50
from .baseline import Baseline
from .baseline_stage3 import Baseline_stage3
from .cls_vit import NonMultiCLSFER
from .cls_vit_stage3 import get_NonMultiCLSFER_stage3, get_RepeatCLSFER, get_RepeatAttentionCLSFER, get_ExpandCLSFER, get_MultiScaleCLSFER, get_NonMultiCLSFER_onlyCLS

def create_model(args, config):
    model = None
    if 'RepVGG' in config.MODEL.ARCH:
        model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint, config=config)
    elif config.MODEL.ARCH == 'IResNet50':
        model = iresnet50(num_features=config.MODEL.NUM_CLASS)
    elif config.MODEL.ARCH == 'CLSFERBaseline':
        model = Baseline(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH, mlp_ratio=config.MODEL.MLP_RATIO,
                        attn_drop=config.MODEL.ATTN_DROP, )
    elif config.MODEL.ARCH == 'CLSFERBaseline_stage3':
        model = Baseline_stage3(num_classes=config.MODEL.NUM_CLASS, depth=config.MODEL.DEPTH, mlp_ratio=config.MODEL.MLP_RATIO,
                         attn_drop=config.MODEL.ATTN_DROP,)
    elif config.MODEL.ARCH == 'NonMultiCLSFER':
        model = NonMultiCLSFER(img_size=config.DATA.IMG_SIZE, 
                               num_classes=config.MODEL.NUM_CLASS, 
                               depth=config.MODEL.DEPTH,
                               mlp_ratio=config.MODEL.MLP_RATIO,
                               attn_drop=config.MODEL.ATTN_DROP,
                               proj_drop=config.MODEL.PROJ_DROP,
                               drop_path=config.MODEL.DROP_PATH,)
    elif config.MODEL.ARCH == 'NonMultiCLSFER_stage3':
        model = get_NonMultiCLSFER_stage3(config)
    elif config.MODEL.ARCH == 'RepeatCLSFER':
        model = get_RepeatCLSFER(config)
    elif config.MODEL.ARCH == 'RepeatAttentionFER':
        model = get_RepeatAttentionCLSFER(config)
    elif config.MODEL.ARCH == 'ExpandCLSFER':
        model = get_ExpandCLSFER(config)
    elif config.MODEL.ARCH == 'MultiScaleCLSFER':
        model = get_MultiScaleCLSFER(config)
    elif config.MODEL.ARCH == 'NonMultiCLSFER_onlyCLS':
        model = get_NonMultiCLSFER_onlyCLS(config)
        
    return model

