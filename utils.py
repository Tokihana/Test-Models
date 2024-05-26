# utils.py
# inline dependencies
import os
import time
# third-party dependencies
import torch
import torch.nn as nn
from thop import profile, clever_format
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f'Loading checkpoint from {config.TRAIN.RESUME}')
    checkpoint = torch.load(config.TRAIN.RESUME, map_location='cpu')
    logger.info(checkpoint.keys())

    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    #logger.info(checkpoint.keys())
    max_acc = 0.0
    if not config.MODE.EVAL and ('optimizer' in checkpoint.keys() and 'lr_scheduler' in checkpoint.keys() and 'epoch' in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f'Loaded checkpoint from {config.TRAIN.RESUME} successfully')
        if 'max_acc' in checkpoint.keys():
            max_acc = checkpoint['max_acc'] 

    del checkpoint
    torch.cuda.empty_cache()
    return max_acc
    
def save_checkpoint(config, model, epoch, max_acc, optimizer, lr_scheduler, logger, is_best=False):
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if not lr_scheduler == None else None,
        'max_acc': max_acc,
        'epoch': epoch,
        'config': config,
    }
    if not os.path.exists(config.SYSTEM.CHECKPOINT):
        os.makedirs(config.SYSTEM.CHECKPOINT)
    if is_best:
        best_path = os.path.join(config.SYSTEM.CHECKPOINT, f'best.pth')
        torch.save(states, best_path)
        logger.info(f'Save checkpoint to {best_path}')
    else:
        save_path = os.path.join(config.SYSTEM.CHECKPOINT, f'epoch_{epoch}.pth')
        torch.save(states, save_path)
        logger.info(f'Save checkpoint to {save_path}')
        
        
def top1_accuracy(output, targets):
    output = torch.max(output, dim=1).values
    correct = output.eq(targets.reshape(-1, 1))
    acc = correct.float().sum() * 100. / targets.size(0)
    return acc

def load_weights(config, model, logger, finetune=False):
    '''
    only load inference weights
    '''
    checkpoint = torch.load(config.TRAIN.RESUME)
    # logger.info(f'Current Best: {checkpoint["max_acc"]}')
    if finetune == True:
        pops = _return_pop_keys(config, checkpoint)
        for pop in pops:
            checkpoint['state_dict'].pop(pop)
    missing, unexcepted = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(f'Missing: {missing},\t Unexcepted: {unexcepted}\t')
    return model

def load_finetune_weights(config, model, logger):
    '''
    assume that you have a pretrained model and need to fine-tune the last output layers
    '''
    checkpoint = torch.load(config.TRAIN.RESUME)
    pops = _return_pop_keys(config, checkpoint)
    for pop in pops:
        checkpoint.pop(pop)
    missing, unexcepted = model.load_state_dict(checkpoint, strict=False)
    for param in model.parameters():
        if param in pops:
            param.require_grad=True
        else:
            param.require_grad=False
    logger.info(f'FINETUNE Missing: {missing},\t Unexcepted: {unexcepted}\t')
    return model

def _return_pop_keys(config, checkpoint):
    if config.MODEL.ARCH == 'RepVGGplus-L2pse':
        pop_keys = [key for key in checkpoint.keys() if 'aux.3' in key or 'linear' in key]
    elif '14x14' in config.MODEL.ARCH:
        pop_keys = ['head.weight', 'head.bias']
    elif config.MODEL.ARCH == 'TransFER':
        pop_keys = ['head.weight', 'head.bias']
    elif config.MODEL.ARCH in ['AC-CAE_single', 'baseline_single']:
        pop_keys = ['head.linear.weight', 'head.linear.bias']
    else:
        pop_keys = None
    return pop_keys
                    
@torch.no_grad()    
def compute_flop_params(config, model, logger):
    model = model.cuda()
    #img = torch.rand((1, 3, 224, 224))
    img = torch.rand((1, 3, 112, 112))
    if 'RepVGG' in config.MODEL.ARCH:
        model.switch_repvggplus_to_deploy()
    flops, params = profile(model, inputs=(img.cuda(),))
    #flops, params = profile(model, inputs=(img,))
    flops, params = clever_format([flops, params], '%.3f')
    logger.info(f'number of parms: {params}\t FLOPs:{flops}')
    del img
    return params, flops
    
@torch.no_grad()
def throughput(model, data_loader, logger):
    '''throughput can be used to ensure batch_size'''
    model = model.cuda()
    model.eval()
    
    num_batchs = len(data_loader)
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda()

        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {int(throughput)} time per step {batch_size/throughput:.2f}")
        logger.info(f"will spend {batch_size/throughput*num_batchs:.2f}s for processing {num_batchs} batch")
        return int(throughput), batch_size/throughput
    
    

@torch.no_grad()
def tSNE(config, model, data_loader):
    from simple_tsne import tsne, momentum_func
    import matplotlib.pyplot as plt
    import numpy as np
    if '14x14' in config.MODEL.ARCH:
        # only contains a linear head
        B, C = len(data_loader.dataset), model.head.in_features
        model.head = torch.nn.Identity()
    elif 'AC-CAE' in config.MODEL.ARCH:
        # contains a SEhead model with a linear head
        B, C = len(data_loader.dataset), model.head.linear.in_features
        model.head.linear = torch.nn.Identity()
    print(f'Examples: {B},\t out channels: {C}')
    
    
    model = model.cuda()
    model.eval()
    
    features = np.zeros((B, C))
    classes = np.zeros((B,))
    loc = 0
    for idx, (images, targets) in enumerate(data_loader):
        batch_size = images.shape[0]
        images = images.cuda()
        out = model(images)
        #print(out.shape)
        
        features[loc: loc+batch_size] = out.cpu().detach().numpy()
        classes[loc: loc+batch_size] = targets.detach().numpy()
        loc = loc + batch_size
    print(features.shape)
    
    low_dim = tsne(
        data=features, # Data is mxn numpy array, each row a point
        n_components=2, # Number of dim to embed to
        perp=30, # Perplexity (higher for more spread out data)
        n_iter=1000, # Iterations to run t-SNE for
        lr=100, # Learning rate
        momentum_fn=momentum_func, # Function returning momentum coefficient, this one is the default update schedule
        pbar=True, # Show progress bar
        random_state=42 # Seed for random initialization
    )
    
    plt.figure()
    plt.scatter(low_dim[:,0], low_dim[:,1], c=classes)
    plt.savefig(os.path.join(config.SYSTEM.EXPERIMENT_PATH, f'{config.DATA.DATASET}_{config.MODEL.ARCH}_t-SNE.png'))
    plt.show()
    

    

@torch.no_grad()
def GradCAM(config, model, img_path):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL import Image
    from torchvision import transforms
    
    if '14x14' in config.MODEL.ARCH:
        target_layers = [model.head]
    elif config.MODEL.ARCH in ['AC-CAE_single', 'baseline_single']:
        target_layers = [model.head.linear]
    
    img = Image.open(img_path)
    img = transforms.Resize(config.DATA.IMG_SIZE)(img)
    img = transforms.ToTensor()(img)
    input_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_img = input_img.unsqueeze(0)