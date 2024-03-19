# linear.py - a very simple MLP model for model save/load testing
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, embed_dim = 256, num_classes=7):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(49, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )
        
    def forward(self, x):
        return self.model(x)
    
def create_Test(args, config):
    model = TestModel(num_classes = config.MODEL.NUM_CLASS)