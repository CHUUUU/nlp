import torch
import torch.nn as nn
import config as c
import numpy as np
from torch.autograd import Variable
import define_func as func

class embedding(nn.Module):
    def __init__(self):
        super(embedding, self).__init__()
        self.word_emb = nn.Embedding(c.vocab_size, c.embedding_dim)
        pos_weight = func.get_sinusoid_encoding_table(c.vocab_size)
        self.pos_emb = nn.Embedding.from_pretrained(pos_weight, freeze=True)

    def forward(self, q):
        q = q.permute(1, 0)  # [seq, batch] -> [batch, seq]
        max_pos = q.size(1)
        position = torch.cumsum(torch.ones(max_pos, dtype=torch.long).cuda(), dim=0)
        # position = tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], device='cuda:0')
        return self.word_emb(q) + self.pos_emb(position)


class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(scaled_dot_product_attention, self).__init__()

    def forward(self, q, k, v):
        d_k = k.size(-1)
        k = k.transpose(-1, -2)
        att = torch.matmul(q, k) / np.sqrt(d_k)
        prob = nn.Softmax(dim=-1)(att)
        score = torch.matmul(prob, v)
        return score

class multihead_att(nn.Module):
    def __init__(self):
        super(multihead_att, self).__init__()
        self.head = 8
        self.hidden_size = 512
        self.dim = int(512 / 8)  # 64
        self.scaled_dot_product_att = scaled_dot_product_attention()

        h2 = 128
        self.dense = nn.Linear(self.hidden_size, h2)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, q, k, v):
        # [256, 21, 512] -> [256, 21, 8, 64]
        q = q.view(c.batch, -1, self.head, self.dim) # view는 split을 대신할 수 있는가
        k = k.view(c.batch, -1, self.head, self.dim)
        v = v.view(c.batch, -1, self.head, self.dim)

        att = self.scaled_dot_product_att(q, k, v) # [256, 21, 8, 64]
        att = att.view(c.batch, -1, self.hidden_size) # view는 concat을 대신할 수 있는가 # [256, 21, 512]
        fc = self.dense(att) # [256, 21, 128]
        fc = self.relu(fc)
        return fc  

class add_norm(nn.Module):
    def __init__(self):
        super(add_norm, self).__init__()
        self.layer_norm = nn.LayerNorm(128)
    def forward(self, residual, output):
        return self.layer_norm(residual + output)

class positionwise_fcn(nn.Module):
    def __init__(self):
        super(positionwise_fcn, self).__init__()
        self.dense_up = nn.Linear(c.hidden_size, c.hidden_size*4)
        self.dense_down = nn.Linear(c.hidden_size*4, c.hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, something):
        fc = self.dense_up(something)
        fc = self.relu(fc)
        fc = self.dense_down(fc)
        return fc

class q_k_v(nn.Module):
    def __init__(self):
        super(q_k_v, self).__init__()
        self.dense = nn.Linear(c.hidden_size, c.hidden_size*4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lookup):
        a = self.dense(lookup)
        a = self.relu(a)  # [seq, batch, dim]
        return a

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.embedding = embedding()
        self.dense_to_128 = nn.Linear(c.embedding_dim, c.factorization_emb_dim)
        self.q = q_k_v()
        self.k = q_k_v()
        self.v = q_k_v()
        self.relu = nn.ReLU(inplace=True)

        self.multi_head = multihead_att()
        self.add_norm = add_norm()
        self.positionwise_fcn = positionwise_fcn()

    def forward(self, q):
        lookup = self.embedding(q)  # [batch, seq, dim] [64, 29, 300]
        lookup = self.dense_to_128(lookup)
        lookup = self.relu(lookup)  # [256, 21, 128]
 
        q = self.q(lookup)  # [256, 21, 512]
        k = self.k(lookup)
        v = self.v(lookup)

        m_h = self.multi_head(q, k, v)  # [256, 21, 128]
        
        something = self.add_norm(lookup, m_h)  # [256, 21, 128]

        p_fcn = self.positionwise_fcn(something)  # [256, 21, 128]

        output = self.add_norm(something, p_fcn)  # [256, 21, 128]

        return output

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.encoder = encoder()
        self.dense_128 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(c.max_seq * c.hidden_size, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, q):
        q = self.encoder(q)
        q = self.dense_128(q)
        q = self.relu(q)
        q = q.view(c.batch, -1)
        classes = self.dense_3(q)
        return classes
