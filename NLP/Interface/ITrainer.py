import torch.nn as nn
from abc import abstractmethod

class ITrainer():
    @abstractmethod
    def __init__(self, model, optim):
        super(ITrainer, self).__init__()
        pass

    @abstractmethod
    def Train(self, data):
        pass
