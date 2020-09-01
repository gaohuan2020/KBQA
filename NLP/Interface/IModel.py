import torch.nn as nn
from abc import abstractmethod

class IModel(nn.Module):
    @abstractmethod
    def __init__(self):
        super(IModel, self).__init__()
        pass

    @abstractmethod
    def forward(self, sentence):
        pass
