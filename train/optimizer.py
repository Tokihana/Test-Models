from torch import optim as optim
from .sam import SAM
from .Esam import ESAM

def build_optimizer(config, model):
    '''
    build optimizer, set weight decay
    '''
    '''
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip = model.no_weight_decay_keywords()
        '''
        
    #parameters = set_weight_decay(model, skip, skip_keywords)
    parameters = model.parameters()
    
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'samadam':
        optimizer = SAM(parameters, base_optimizer=optim.Adam, rho=config.TRAIN.OPTIMIZER.RHO, adaptive=False, 
                        lr=config.TRAIN.BASE_LR, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,)
    elif opt_lower == 'samsgd':
        optimizer = SAM(parameters, base_optimizer=optim.SGD, rho=config.TRAIN.OPTIMIZER.RHO, adaptive=False,
                        lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'esamadam':
        base_opt = optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                             weight_decay=config.TRAIN.WEIGHT_DECAY)
        optimizer = ESAM(parameters, base_opt, rho=config.TRAIN.OPTIMIZER.RHO, adaptive=False)
        
    else:
        raise f'not supported optim {opt_lower}'
    if 'esam' in opt_lower:
        return optimizer, base_opt
    else:
        return optimizer

def set_weight_decay(model, skip_list, skip_keywords):
    '''
    ensure no decay was applied on frozen and skip param 
    '''
    has_decay=[]
    no_decay=[]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_key_words_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params':has_decay},
            {'params':no_decay, 'weight_decay':0.}]

def check_key_words_in_name(name, keywords=()):
    isin=False
    for keyword in keywords:
        if keyword in name:
            isin=True
            
    return isin