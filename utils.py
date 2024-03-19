# utils.py
# inline dependencies
import os
import time
# third-party dependencies
import torch
from thop import profile, clever_format

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
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_acc': max_acc,
        'epoch': epoch,
        'config': config,
    }
    if is_best:
        best_path = os.path.join(config.SYSTEM.CHECKPOINT, 'best.pth')
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

def load_weights(config, model, logger):
    '''
    only load inference weights
    '''
    checkpoint = torch.load(config.TRAIN.RESUME)
    missing, unexcepted = model.load_state_dict(checkpoint, strict=False)
    logger.info(f'FINETUNE Missing: {missing},\t Unexcepted: {unexcepted}\t')
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
    else:
        pop_keys = None
    return pop_keys
                    
@torch.no_grad()    
def compute_flop_params(config, model, logger):
    img = torch.rand((1, 3, 224, 224))
    if 'RepVGG' in config.MODEL.ARCH:
        model.switch_repvggplus_to_deploy()
    flops, params = profile(model, inputs=(img.cuda(),))
    flops, params = clever_format([flops, params], '%.3f')
    logger.info(f'number of parms: {params}\t FLOPs:{flops}')
    del img
    
@torch.no_grad()
def throughput(model, data_loader, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)

        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return