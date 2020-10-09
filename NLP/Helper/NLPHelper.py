import torch
from NLP.Config.Setting import *

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def CheckKeyInConfig(key, config, type):
    if key in config:
        if type != None or type != "":
            if isinstance(config[key], type):
                return True
    return False

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_batch(data ,word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad=[]
    tags_pad=[]
    for seq,tag in zip(seqs, tags):
        seq_pad = seq + [PAD_TAG] * (max_len-len(seq))
        tag_pad = tag + [PAD_TAG] * (max_len-len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad

def word_to_idx(training_data):
    word_to_ix = {}
    word_to_ix[PAD_TAG] = 0
    for sentence, _ in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix