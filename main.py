# main.py
# inline dependencies
import time
import datetime
import argparse
# third-party dependencies
import wandb
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

    parser.add_argument('--config', default='./config/yaml/FERPlus_CLSFERBaseline_Stage3.yaml', type=str, help='path to config yaml')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def main():
    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.SYSTEM.PROJECT_NAME,
    
        # track hyperparameters and run metadata
        config={
            # emperiment name
            "experiment name": config.SYSTEM.EXPERIMENT_NAME,
            # dataset
            "dataset": config.DATA.DATASET,
            "batch size": config.DATA.BATCH_SIZE,
            "img size": config.DATA.IMG_SIZE,
            "num workers": config.DATA.NUM_WORKERS,
            "pin memory": config.DATA.PIN_MEMORY,
            # model architecture
            "architecture": config.MODEL.ARCH,
            "encoder depth": config.MODEL.DEPTH,
            "num classes": config.MODEL.NUM_CLASS,
            "mlp ratio": config.MODEL.MLP_RATIO,
            "attn drop": config.MODEL.ATTN_DROP,
            # training settings
            "learning_rate": config.TRAIN.BASE_LR,
            "epochs": config.TRAIN.EPOCHS,
            "resume": config.TRAIN.RESUME,
            ## criterion
            "criterion": config.TRAIN.CRITERION.NAME,
            "LabelSmoothing": config.TRAIN.CRITERION.LABEL_SMOOTHING,
            ## optimizer
            "optimizer": config.TRAIN.OPTIMIZER.NAME,
            "optimizer eps": config.TRAIN.OPTIMIZER.EPS,
            "optimizer beta1": config.TRAIN.OPTIMIZER.BETAS[0],
            "optimizer beta2": config.TRAIN.OPTIMIZER.BETAS[1],
            "optimizer momentum": config.TRAIN.OPTIMIZER.MOMENTUM,
            ## lr scheduler
            "scheduler": config.TRAIN.LR_SCHEDULER.NAME,
            "scheduler gamma": config.TRAIN.LR_SCHEDULER.GAMMA,
            # augment
            "mixup alpha": config.AUG.MIXUP,
        }
    )
    
    # load datasets
    train_loader, val_loader, mix_fn = build_loader(config)
    logger.info(f'Train: {len(train_loader)}, Test: {len(val_loader)}')
    
    # create model, move to cuda, log parameters, flops
    model = create_model(args, config)
    model.cuda()
    
    # logger flop and params
    if config.MODE.FLOP:
        params, flops = compute_flop_params(config, model, logger)
        wandb.log({'params':params, 'flops':flops})
        return
    
    if config.MODE.THROUGHPUT:
        through, step_time=throughput(model, train_loader, logger)
        wandb.log({'throughput':through, 'time per stem':step_time})
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
        if epoch % config.SYSTEM.SAVE_FREQ == 0 or epoch >= (config.TRAIN.EPOCHS-1):
            save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
        if val_loader is not None:
            acc, loss = validate(config, model, val_loader, logger)
            wandb.log({'acc':acc, 'loss':loss})
            max_acc = max(max_acc, acc)
            wandb.log({'running max acc':max_acc})
            logger.info(f'Epoch: [{epoch}/{config.TRAIN.EPOCHS}], Acc: {acc:.3f}%, Max: {max_acc:.3f}%')
            if max_acc == acc: # max_acc updated
                save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger, is_best=True)
                  
    wandb.log({'max acc':max_acc})
    total_time = time.time() - start_time
    logger.info(f'Total training time: {str(datetime.timedelta(seconds=int(total_time)))}')
    
    # release cache
    torch.cuda.empty_cache()
    
    wandb.finish()
    
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
        if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
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
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    if config.TRAIN.LR_SCHEDULER.NAME == 'exponential':
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
                  
def _test_lr():
    #lr_list = [0.1**i for i in range(1, 8)]
    lr_list = (0.1**torch.linspace(4.5, 5.5, 7)).tolist()
    check_path = config.SYSTEM.CHECKPOINT
    for lr in lr_list:
        config.defrost()
        config.SYSTEM.CHECKPOINT = f'{check_path}_lr{str(lr)}'
        config.TRAIN.BASE_LR = lr
        config.freeze()
        main()
        
def _test_eps():
    eps_list = [0.1**i for i in range(6, 11)] # 6, 7, 8, 9, 10
    config.defrost()
    config.TRAIN.EPOCHS = 20
    config.freeze()
    for eps in eps_list:
        config.defrost()
        config.TRAIN.OPTIMIZER.EPS = eps
        config.freeze()
        main()
        
def _test_beta1():
    beta1_list = [0.7, 0.8, 0.85, 0.9, 0.95, 1]
    config.defrost()
    config.TRAIN.EPOCHS = 25
    config.freeze()
    for beta1 in beta1_list:
        config.defrost()
        config.TRAIN.OPTIMIZER.BETAS = (beta1, config.TRAIN.OPTIMIZER.BETAS[1])
        config.freeze()
        main()
        
def _test_beta2():
    beta2_list = [0.8, 0.9, 0.99, 0.995, 0.997, 0.999, 0.999, 1]
    config.defrost()
    config.TRAIN.EPOCHS = 35
    config.freeze()
    for beta2 in beta2_list:
        config.defrost()
        config.TRAIN.OPTIMIZER.BETAS = (config.TRAIN.OPTIMIZER.BETAS[0], beta2)
        config.freeze()
        main()

def _test_gamma():
    gamma_list = [0.955, 0.98]
    config.defrost()
    config.TRAIN.EPOCHS = 100
    config.freeze()
    for gamma in gamma_list:
        config.defrost()
        config.TRAIN.LR_SCHEDULER.GAMMA = gamma
        config.freeze()
        main()
        
def _test_mixup():
    alpha_list = [0., 0.1, 0.2, 0.6, 1.]
    for alpha in alpha_list:
        config.defrost()
        config.AUG.MIXUP = alpha
        config.freeze()
        main()
        
def _test_drop_attn():
    drops = [0., 0.1, 0.2, 0.4, 0.6, 0.8]
    for drop in drops:
        config.defrost()
        config.MODEL.ATTN_DROP = drop
        config.freeze()
        main()

    
if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger(config.SYSTEM.LOG, name='testlog.log')
    # main()
    _test_lr()
    #_test_gamma()     
    #_test_mixup()
    #_test_drop_attn()
    