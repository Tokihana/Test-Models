import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchsampler import ImbalancedDatasetSampler

def build_loader(config):
    config.defrost()
    train_dataset, val_dataset, config.MODEL.NUM_CLASS = build_dataset(config=config)
    config.freeze()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               # shuffle=True, # may not need to specified when use sampler
                                               batch_size=config.DATA.BATCH_SIZE,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=config.DATA.BATCH_SIZE,
                                             num_workers=config.DATA.NUM_WORKERS,
                                             pin_memory=config.DATA.PIN_MEMORY,
                                             drop_last=False)

    cutmix = v2.CutMix(num_classes=config.MODEL.NUM_CLASS)
    mixup = v2.MixUp(num_classes=config.MODEL.NUM_CLASS)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup]) # used after loader sampling
                                                        # example:
                                                        # for (images, targets) in loader:
                                                            # images, targets = mixup(images, targets)
    
    return train_loader, val_loader, cutmix_or_mixup

def build_dataset(config):
    # RAF-DB
    if config.DATA.DATASET == 'RAF-DB':
        train_transform, val_transform = _get_rafdb_transform()
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'AffectNet_7':
        train_transform, val_transform = _get_affectnet_transform()
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'AffectNet_8':
        train_transform, val_transform = _get_affectnet_transform()
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 8
    elif config.DATA.DATASET == 'FERPlus':
        train_transform, val_transform = _get_affectnet_transform()
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 8
    elif config.DATA.DATASET == 'MiniTest': # minitest is a very, very small subset of RAF-DB
        train_transform, val_transform = _get_rafdb_transform()
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    else:
        raise NotImplementError("DATASET NOT SUPPORTED")
    return train_dataset, val_dataset, nb_classes

def _get_rafdb_transform():
    train_transform = v2.Compose([
        v2.Resize((224, 224)), # Resize() only accept PIL or Tensor type images
        v2.RandomHorizontalFlip(),
        #v2.ToTensor(), # warned by Torch: ToTensor() will be removed in a future release, use [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] instead
                        # if the param scale is True, the range of the inputs will be normalized
                        # note that ToImage() only accept the inputs with legth 3
        #v2.Normalize(mean, std),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(scale=(0.02, 0.1)),
    ])
    val_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, val_transform

def _get_affectnet_transform():
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.RandomErasing(scale=(0.02, 0.1)),
    ])
    val_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, val_transform

def _get_ferplus_transform():
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.RandomErasing(scale=(0.02, 0.1)),
    ])
    val_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, val_transform