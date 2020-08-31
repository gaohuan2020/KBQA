import torch
import torch.nn as nn
from NLP.Interface.IModel import IModel
from NLP.Config.Setting import LSTMCRFModelConfig
from NLP.Helper.NLPHelper import CheckKeyInConfig

class LSTM(IModel):
    def __init__(self):
        super().__init__()
        self.embeddingDim = 100
        self.vocabSize = 100
        self.hiddenDim = 100
        self.tagSize = 10
        if CheckKeyInConfig("embedDim", LSTMCRFModelConfig, int):
            self.embeddingDim = LSTMCRFModelConfig["embedDim"]
        if CheckKeyInConfig("vocabSize", LSTMCRFModelConfig, int):
            self.vocabSize = LSTMCRFModelConfig["vocabSize"]
        if CheckKeyInConfig("hiddenDim", LSTMCRFModelConfig, int):
            self.hiddenDim = LSTMCRFModelConfig["hiddenDim"]
        if CheckKeyInConfig("tagSize", LSTMCRFModelConfig, int):
            self.hiddenDim = LSTMCRFModelConfig["tagSize"]
        #TODO check the value of all parameters in LSTMCRF model. if the value is invalid return None
        self.wordEmbedding = nn.Embedding(self.vocabSize, self.embeddingDim)
        self.lstm = nn.LSTM(self.embeddingDim, self.hiddenDim // 2, num_layers=1, bidirectional= True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hiddenDim, self.tagSize) 
        self.transitions = nn.Parameter(torch.randn(self.tagSize, self.tagSize))

    def init_hidden(self):
        return (torch.randn(2, 1, self.hiddenDim // 2),
                torch.randn(2, 1, self.hiddenDim // 2))
    
    def forward(self, sentence):
        pass