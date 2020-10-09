import torch.nn as nn
from abc import abstractmethod

class ITester():
    @abstractmethod
    def __init__(self, model, optim):
        super(ITester, self).__init__()
        pass

    @abstractmethod
    def Test(self, data):
        pass

    @abstractmethod
    def Predict(self, data):
        pass
