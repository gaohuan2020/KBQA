import unittest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from NLP.Helper.NLPHelper import *
from NLP.Model.EntityRelationReorganization.Model import LSTMCRF
from NLP.Model.DataSet import NERDataSet
from NLP.Model.Train import Trainer
from NLP.Model.Test import Tester
from NLP.Config.Setting import *
from tqdm import trange

class LSTMCRFModel_test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.training_data = [(
            "the wall street journal reported today that apple corporation made money".split(),
            "B I I I O O O B I O O".split()
        ), (
            "georgia tech is a university in georgia".split(),
            "B I O O O O B".split()
        )]
        self.word_to_ix = word_to_idx(self.training_data)
        self.tag_to_ix = Sequence_TAG
        # Make up some training data
        self.model = LSTMCRF(len(self.word_to_ix), self.tag_to_ix)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

    def test_lstmCRFModel(self):
        with torch.no_grad():
            precheck_sent = prepare_sequence(self.training_data[0][0], self.word_to_ix)
            precheck_tags = torch.tensor([self.tag_to_ix[t] for t in self.training_data[0][1]], dtype=torch.long)
            predict = self.model(precheck_sent)
            self.assertEqual(len(predict), 2, 2)
            self.assertEqual(len(predict[1]), len(precheck_tags), len(precheck_tags))

    @classmethod
    def test_lstmCRFModelTrainAndTest(self):
        startLoss = 10000
        loss = 10000
        for i in trange(100):
            self.model.zero_grad()
            sentence_in_pad, targets_pad = prepare_sequence_batch(self.training_data, self.word_to_ix, self.tag_to_ix)
            loss = self.model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
            loss.backward()
            if i == 0:
                startLoss = loss.item()
            self.optimizer.step()
        self.assertGreater(startLoss, startLoss, loss.item())

        # Check predictions after training
        with torch.no_grad():
            precheck_sent  = prepare_sequence(self.training_data[0][0], self.word_to_ix)
            precheck_tags = torch.tensor([self.tag_to_ix[t] for t in self.training_data[0][1]], dtype=torch.long)
            predict = self.model(precheck_sent)

        
    @classmethod
    def test_dataLoaderAndTrainer(self):
        train_dataset = NERDataSet(self.word_to_ix,self.tag_to_ix, self.training_data)
        training_data_loader = DataLoader(train_dataset,
                                            batch_size=LSTMCRFModelConfig['batchSize'],
                                            num_workers=LSTMCRFModelConfig['threads'],
                                            shuffle=LSTMCRFModelConfig['isShuffle'])
        for data, tag in training_data_loader:
            print(data)
            print(tag)
        
        trainer = Trainer(self.model, self.optimizer, training_data_loader)
        trainer.Train()