from NLP.Helper.NLPHelper import word_to_idx
from torch.utils import data
from NLP.Helper.NLPHelper import *


class NERDataSet(data.Dataset):
    def __init__(self, word_to_idx, tag_to_idx, data):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.data = data

    def __getitem__(self, index):
        sentence_in_pad, targets_pad = prepare_sequence_batch(
            self.data, self.word_to_idx, self.tag_to_idx)
        return sentence_in_pad[index], targets_pad[index]

    def __len__(self):
        return len(self.data)