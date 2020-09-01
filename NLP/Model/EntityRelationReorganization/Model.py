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
        self.tagSize = 6
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

    def get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentenceEmbeddings = self.wordEmbedding(sentence)
        lstm_out, self.hidden = self.lstm(sentenceEmbeddings)
        lstm_feature = self.hidden2tag(lstm_out)
        return lstm_feature

    def forward_inference_alg(self, features):
        init_alphas = torch.full([features.shape[0], self.tagSize], -10000.)
        init_alphas[:, 3] = 0.
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(features.shape[1]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * features.shape[2]).transpose(0, 1)
            t_r1_k = torch.unsqueeze(features[:, feat_index, :], 1).transpose(1, 2)  # +1
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[4].repeat([features.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def score_sentence(self,features, tag):
        score = torch.zeros(tag.shape[0])#.to('cuda')
        tag = torch.cat([torch.full([tag.shape[0],1],3).long(),tag],dim=1)
        for i in range(features.shape[1]):
            feat=features[:,i,:]
            score = score + \
                    self.transitions[tag[:,i + 1], tag[:,i]] + feat[range(feat.shape[0]),tag[:,i + 1]]
        score = score + self.transitions[4, tag[:,-1]]
        return score

    def neg_log_likelihood(self, sentence, tag):
        lstm_feature = self.get_lstm_features(sentence)
        alpha = self.forward_inference_alg(lstm_feature)
        gold_score = self.score_sentence(lstm_feature, tag)[0]
        return alpha - gold_score

    def forward(self, sentence):
        lstm_feature = self.get_lstm_features(sentence)
        alpha = self.forward_inference_alg(lstm_feature)
        return alpha
    #TODO add viterbi algorithm for inference