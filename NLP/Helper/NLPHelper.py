import torch

def CheckKeyInConfig(key, config, type):
    if key in config:
        if type != None or type != "":
            if isinstance(config[key], type):
                return True
    return False

def prepare_sequence_batch(data ,word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad=[]
    tags_pad=[]
    for seq,tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len-len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len-len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad