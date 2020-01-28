import torch
import torch.nn as nn
import config as config
import numpy as np
from torch.autograd import Variable
import define_func as func

class embedding(nn.Module):
    def __init__(self, vocab_size):
        super(embedding, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, config.embedding_dim)
        pos_weight = func.get_sinusoid_encoding_table(vocab_size)
        self.pos_emb = nn.Embedding.from_pretrained(pos_weight, freeze=True)
        
    def forward(self, sentence):
        sentence = sentence.permute(1, 0) # [21, 256] -> [256, 21]
        max_pos = sentence.size(1)
        position = torch.cumsum(torch.ones(max_pos, dtype=torch.long).to(config.gpu), dim=0)
        return self.word_emb(sentence) + self.pos_emb(position)

    
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
        self.dim = int(512 / 8) # 64
        self.scaled_dot_product_att = scaled_dot_product_attention()

        h2 = 128
        self.dense = nn.Linear(self.hidden_size, h2)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, q, k, v):
        # [256, 21, 512] -> [256, 21, 8, 64]
        q = q.view(config.batch, -1, self.head, self.dim) 
        k = k.view(config.batch, -1, self.head, self.dim)
        v = v.view(config.batch, -1, self.head, self.dim)

        att = self.scaled_dot_product_att(q, k, v) # [256, 21, 8, 64]
        att = att.view(config.batch, -1, self.hidden_size) # [256, 21, 512]
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
        self.dense_up = nn.Linear(128, 512)
        self.dense_down = nn.Linear(512, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, something):
        fc = self.dense_up(something)
        fc = self.relu(fc)
        fc = self.dense_down(fc)
        return fc

class q_k_v(nn.Module):
    def __init__(self):
        super(q_k_v, self).__init__()
        emb_size = 128
        hidden_size = 512
        self.dense = nn.Linear(emb_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lookup):
        a = self.dense(lookup)
        a = self.relu(a) # [seq, batch, dim]
        return a

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super(encoder, self).__init__()
        self.embedding = embedding(vocab_size)
        
        self.q = q_k_v()
        self.k = q_k_v()
        self.v = q_k_v()
        self.relu = nn.ReLU(inplace=True)

        self.multi_head = multihead_att()
        self.add_norm = add_norm()
        self.positionwise_fcn = positionwise_fcn()

    def forward(self, sentence):
        lookup = self.embedding(sentence) # 300 dim # [batch, seq, dim] [256, 21, 128]
 
        q = self.q(lookup) # [256, 21, 512]
        k = self.k(lookup)
        v = self.v(lookup)

        m_h = self.multi_head(q, k, v) # [256, 21, 128]
        
        norm = self.add_norm(lookup, m_h) # [256, 21, 128]

        p_fcn = self.positionwise_fcn(norm) # [256, 21, 128]

        output = self.add_norm(norm, p_fcn) # [256, 21, 128]

        return output

class classifier(nn.Module):
    def __init__(self, vocab_size):
        super(classifier, self).__init__()
        self.encoder = encoder(vocab_size)
        self.relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(config.max_seq * 128, config.output_class)
        
    def forward(self, sentence):
        encoding = self.encoder(sentence) # [256, 21, 128]
        encoding = self.relu(encoding)
        fcn = encoding.view(config.batch, -1) # [256, 21 * 128]
        print(fcn.shape)
        output = self.dense(fcn) # [256, classes]
        print(output.shape)
        return output
