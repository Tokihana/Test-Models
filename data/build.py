import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from .dataset_raf import RafDataSet
from .dataset_ferplus import FERDataSet

def build_loader(config):
    config.defrost()
    train_dataset, val_dataset, config.MODEL.NUM_CLASS = build_dataset(config=config)
    config.freeze()
    if config.DATA.IMBALANCED_SAMPLER:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               # shuffle=True, # not need to specified when use sampler
                                               batch_size=config.DATA.BATCH_SIZE,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               drop_last=True)
    else: # not config.DATA.IMBALANCED_SAMPLER
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle=True, # not need to specified when use sampler
                                               batch_size=config.DATA.BATCH_SIZE,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=config.DATA.VAL_BATCH_SIZE,
                                             num_workers=config.DATA.NUM_WORKERS,
                                             pin_memory=config.DATA.PIN_MEMORY,
                                             drop_last=False)

    if config.AUG.MIXUP > 0.:
        cutmix = v2.CutMix(alpha=config.AUG.MIXUP, num_classes=config.MODEL.NUM_CLASS)
        mixup = v2.MixUp(alpha=config.AUG.MIXUP, num_classes=config.MODEL.NUM_CLASS)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup]) # used after loader sampling
                                                        # example:
                                                        # for (images, targets) in loader:
                                                            # images, targets = mixup(images, targets)
    else:
        cutmix_or_mixup = None
    
    return train_loader, val_loader, cutmix_or_mixup

def build_dataset(config):
    img_size = config.DATA.IMG_SIZE
    if config.DATA.DATASET == 'RAF-DB':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'AffectNet_7':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'AffectNet_8':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 8
    elif config.DATA.DATASET == 'FERPlus':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'Training'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'PublicTest'), val_transform)
        nb_classes = 8
    elif config.DATA.DATASET == 'MiniTest': # minitest is a very, very small subset of RAF-DB
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'CK+':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 8
    elif config.DATA.DATASET == 'JAFFE':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'SFEW':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'test'), val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'dataseted-RAF-DB':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = RafDataSet(config.DATA.DATA_PATH, train=True, transform=train_transform, basic_aug=True)
        val_dataset = RafDataSet(config.DATA.DATA_PATH, train=False, transform=val_transform)
        nb_classes = 7
    elif config.DATA.DATASET == 'dataseted-FERPlus':
        train_transform, val_transform = _get_transform(config, img_size)
        train_dataset = FERDataSet(config.DATA.DATA_PATH, train=True, transform=train_transform, basic_aug=True)
        val_dataset = FERDataSet(config.DATA.DATA_PATH, train=False, transform=val_transform)
        nb_classes = 8
    else:
        raise NotImplementError("DATASET NOT SUPPORTED")
    return train_dataset, val_dataset, nb_classes

def _get_transform(config, img_size=224):
    train_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=config.DATA.ERASING_P, scale=config.DATA.ERASING_SCALE),
    ])
    val_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def _get_rafdb_transform(img_size=224):
    train_transform = v2.Compose([
        v2.Resize((img_size, img_size)), # Resize() only accept PIL or Tensor type images
        v2.RandomHorizontalFlip(),
        #v2.ToTensor(), # warned by Torch: ToTensor() will be removed in a future release, use [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] instead
                        # if the param scale is True, the range of the inputs will be normalized
                        # note that ToImage() only accept the inputs with legth 3
        #v2.Normalize(mean, std),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(scale=config.DATA.ERASING_SCALE),
    ])
    val_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def _get_affectnet_transform(img_size=224):
    train_transform = v2.Compose([
        transforms.ToPILImage(),
        v2.Resize((img_size, img_size)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.RandomErasing(p=1., scale=config.DATA.ERASING_SCALE),
    ])
    val_transform = v2.Compose([
        transforms.ToPILImage(),
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, val_transform

def _get_ferplus_transform(img_size=224):
    train_transform = v2.Compose([
        transforms.ToPILImage(),
        v2.Resize((img_size, img_size)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.RandomErasing(p=1., scale=config.DATA.ERASING_SCALE),
    ])
    val_transform = v2.Compose([
        transforms.ToPILImage(),
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, val_transform