import unittest
import torch
from NLP.Helper.NLPHelper import prepare_sequence_batch
from NLP.Model.EntityRelationReorganization.Model import LSTM


class LSTMCRFModel_test(unittest.TestCase):
    def test_lstmCRFModel(self):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        PAD_TAG = "<PAD>"
        training_data = [(
            "the wall street journal reported today that apple corporation made money"
            .split(), "B I I I O O O B I O O".split()),
                         ("georgia tech is a university in georgia".split(),
                          "B I O O O O B".split())]
        word_to_ix = {}
        word_to_ix['<PAD>'] = 0
        for sentence, tags in training_data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        tag_to_ix = {
            "B": 0,
            "I": 1,
            "O": 2,
            START_TAG: 3,
            STOP_TAG: 4,
            PAD_TAG: 5
        }
        lstm = LSTM()
        with torch.no_grad():
            precheck_sent, precheck_lable = prepare_sequence_batch(
                training_data, word_to_ix, tag_to_ix)
            lstm.neg_log_likelihood(precheck_sent, precheck_lable)
            lstm(precheck_sent)
