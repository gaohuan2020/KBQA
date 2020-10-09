from tqdm import trange
from NLP.Config.Setting import *

class Trainer():
    def __init__(self, model, optim, data):
        self.model = model
        self.optim = optim
        self.data = data

    def Train(self):
        for sentence_in_pad, targets_pad in self.data:
            for i in trange(LSTMCRFModelConfig['epoch']):
                self.model.zero_grad()
                loss = self.model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
                loss.backward()
                self.optim.step()

