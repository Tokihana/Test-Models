import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def build_criterion(config):
    criterion = None
    if config.TRAIN.CRITERION.NAME == 'SoftTargetCE':
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.TRAIN.CRITERION.NAME == 'LabelSmoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=config.TRAIN.CRITERION.LABEL_SMOOTHING)
    elif config.TRAIN.CRITERION.NAME == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config.TRAIN.CRITERION.NAME == 'POSTERCrossEntropy':
        criterion = POSTERCrossEntropy(smoothing=config.TRAIN.CRITERION.LABEL_SMOOTHING)
    return criterion

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1).long()) # gather() requires index as type int64, so we need to true target to long int type
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class POSTERCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(POSTERCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.CE = torch.nn.CrossEntropyLoss()
        self.LSCE = LabelSmoothingCrossEntropy(smoothing)
        
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 2 * self.LSCE(x, target) + self.CE(x, target)