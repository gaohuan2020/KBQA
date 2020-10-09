from NLP.Helper.NLPHelper import word_to_idx
import torch

class Tester():
    def __init__(self, model):
        self.model = model

    def Test(self, data):
        for sentence, targets in data:
            with torch.no_grad():
                predict = self.model.forward_parallel(sentence)

    def Eval(self, data):
        with torch.no_grad():
            predict = self.model(data)
            return predict
        
