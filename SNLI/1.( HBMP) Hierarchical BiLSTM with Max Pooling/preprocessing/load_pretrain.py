import config as config
import torch
import os
import numpy as np

def get_pretrain_embedding(vocab): # list
    if os.path.exists(config.glove_npy):
        embedding = np.load(config.glove_npy)
        embedding = torch.FloatTensor(embedding)
        print("pretrain embedding 을 불러왔다.")
        return embedding
    elif os.path.exists(config.glove):
        print("Read glove.840B.300d.txt ...")
        pretrain_emb = {}
        with open(config.glove, encoding="utf-8") as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                pretrain_emb[word] = coefs
        vocab_size = len(vocab)
        embedding = np.zeros((vocab_size, 300))

        missing_words = []
        for n, word in enumerate(vocab):
            vector = pretrain_emb.get(word)
            if vector is not None:
                embedding[n] = vector
            else:
                missing_words.append(word)
                print('Missing from GloVe: {}'.format(word))
        print("total missing word length from GloVe : ", len(missing_words))
        embedding = torch.FloatTensor(embedding)
        np.save(config.glove_npy, embedding)
        print("pretrain embedding 을 저장되었다..")        
        return embedding
    else:
        print("missing glove.840B.300d.txt")

    