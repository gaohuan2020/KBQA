import unittest
from NLP.Model.EntityRelationReorganization.Model import LSTM

class LSTMCRFModel_test(unittest.TestCase):
    def test_lstmCRFModel(self):
        lstm = LSTM()
