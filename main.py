# main.py
# inline dependencies
import time
import datetime
import argparse
# third-party dependencies
import torch
import torch.nn as nn
from timm.utils import accuracy, AverageMeter
# local dependencies
from config.config import get_config
from data.build import build_loader
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from utils import save_checkpoint, load_checkpoint, top1_accuracy, load_finetune_weights, compute_flop_params, throughput
from model import create_model

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/RAF-DB_RepVGGplus-L2pse_FINETUNE.yaml', type=str, help='path to config yaml')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def main():
    # load datasets
    train_loader, val_loader, mix_fn = build_loader(config)
    logger.info(f'Train: {len(train_loader)}, Test: {len(val_loader)}')
    
    # create model, move to cuda, log parameters, flops
    model = create_model(args, config)
    model.cuda()
    
    # logger flop and params
    if config.MODE.FLOP:
        compute_flop_params(config, model, logger)
        return
    
    if config.MODE.THROUGHPUT:
        throughput(model, val_loader, logger)
        return
    
    # build optimier
    optimizer = build_optimizer(config, model)
    logger.info(f"LR: {optimizer.param_groups[0]['lr']}")
    
    # check if EVAL MODE or some other running MODE you need, such as THROUGHPUT_MODE
    if config.MODE.EVAL:
        load_weights(config, model, loggr)
        acc, loss = validate(config, model, val_loader, logger)
        logger.info(f'Epoch: [{epoch}/{config.TRAIN.EPOCHS}], Acc: {acc:.3f}%, Max: {max_acc:.3f}%')
        return
    
    if config.MODE.FINETUNE:
        model = load_finetune_weights(config, model, logger)
    
    # build scheduler
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    
    # init criterion
    criterion = build_criterion(config)
        
    # whether needs to resume model?
    max_acc = 0.0
    if not config.MODE.FINETUNE and not config.TRAIN.RESUME == '': # training time model resume
        max_acc = load_checkpoint(config=config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
    
    # start training
    logger.info(f'Start training')
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config=config, model=model, data_loader=train_loader, epoch=epoch, mix_fn = mix_fn, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
        if epoch % config.SYSTEM.SAVE_FREQ == 0 or epoch >= (config.TRAIN.EPOCHS-5):
            save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
        if val_loader is not None:
            acc, loss = validate(config, model, val_loader, logger)
            max_acc = max(max_acc, acc)
            logger.info(f'Epoch: [{epoch}/{config.TRAIN.EPOCHS}], Acc: {acc:.3f}%, Max: {max_acc:.3f}%')
            if max_acc == acc: # max_acc updated
                save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger, is_best=True)
    
    total_time = time.time() - start_time
    logger.info(f'Total training time: {str(datetime.timedelta(seconds=int(total_time)))}')
    
def train_one_epoch(config, model, data_loader, criterion, optimizer, lr_scheduler, epoch, mix_fn, logger):
    model.train()
    optimizer.zero_grad() # clear accumulated gradients
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda()
        targets = targets.cuda()
        
        if mix_fn is not None:
            images, targets = mix_fn(images, targets)
            
        outputs = model(images)
        
        if type(outputs) is dict: # for RepVGGplus-L2pse
            loss = 0.0
            for name, pred in outputs.items():
                if 'aux' in name:
                    loss += 0.1*criterion(pred, targets)
                else:
                    loss += criterion(pred, targets)
        else:
            loss = criterion(outputs, targets) 
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER == 'cosine':
            lr_scheduler.step_update(epoch*num_steps*idx)
        
        loss_avg.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.SYSTEM.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg*(num_steps - idx) # estimated time of arrival
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{len(data_loader)}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    if config.TRAIN.LR_SCHEDULER == 'exponential':
        lr_scheduler.step()
    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
        
    
@torch.no_grad()
def validate(config, model, data_loader, logger):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    batch_avg = AverageMeter()
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda()
        targets = targets.cuda()
        
        if config.MODEL.ARCH == 'RepVGGplus-L2pse':
            output = model(images)['main']
        else:
            output = model(images)
            
        loss = criterion(output, targets)
        # acc = top1_accuracy(output, targets)
        acc = accuracy(output, targets, topk=(1, ))[0]
        loss_avg.update(loss.item(), targets.size(0))
        acc_avg.update(acc.item(), targets.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.SYSTEM.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})\t'
                f'Acc {acc_avg.val:.3f} ({acc_avg.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc {acc_avg.avg:.3f}')
    return acc_avg.avg,  loss_avg.avg
    
if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    main()
    
    