import unittest
import torch
import torch.optim as optim
from NLP.Helper.NLPHelper import *
from NLP.Model.EntityRelationReorganization.Model import LSTMCRF
from NLP.Config.Setting import *

class LSTMCRFModel_test(unittest.TestCase):
    def test_lstmCRFModel(self):
        # Make up some training data
        training_data = [(
            "the wall street journal reported today that apple corporation made money".split(),
            "B I I I O O O B I O O".split()
        ), (
            "georgia tech is a university in georgia".split(),
            "B I O O O O B".split()
        )]

        word_to_ix = {}
        word_to_ix[PAD_TAG] = 0
        for sentence, tags in training_data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}

        model = LSTMCRF(len(word_to_ix), tag_to_ix)
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Check predictions before training
        with torch.no_grad():
            precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
            print(model(precheck_sent))

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our batch inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # If training_data can't be included in one batch, you need to sample them to build a batch
            sentence_in_pad, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            print(loss.item())
            optimizer.step()

        # Check predictions after training
        with torch.no_grad():
            precheck_sent  = prepare_sequence(training_data[0][0], word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
            print(model(precheck_sent))
            print(precheck_tags)
            # We got it!
