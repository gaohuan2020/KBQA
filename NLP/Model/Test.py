from NLP.Helper.NLPHelper import word_to_idx
import torch

class Tester():
    def __init__(self, model, data, word_to_idx):
        self.model = model
        self.data = data
        self.word_to_idx = word_to_idx

    def Eval(self, data):
        for sentence, targets in self.data:
            with torch.no_grad():
                predict = self.model(sentence)
