from tqdm import trange
from NLP.Config.Setting import *
from NLP.Interface.ITrainer import ITrainer

class Trainer(ITrainer):
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim

    def Train(self, data):
        for sentence_in_pad, targets_pad in data:
            for i in trange(LSTMCRFModelConfig['epoch']):
                self.model.zero_grad()
                loss = self.model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
                loss.backward()
                self.optim.step()

