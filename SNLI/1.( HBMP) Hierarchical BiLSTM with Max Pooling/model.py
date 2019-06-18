import torch
import torch.nn as nn
import config as config
from torch.autograd import Variable
from preprocessing.load_pretrain import get_pretrain_embedding

class Sentence_Encoder(nn.Module):
    def __init__(self, vocab_list):
        super(Sentence_Encoder, self).__init__()
        self.embedding = nn.Embedding(len(vocab_list), config.embedding_dim)
        if config.use_glove:
            weight = get_pretrain_embedding(vocab_list)
            self.embedding = nn.Embedding.from_pretrained(weight)
            print("glove embedding : ", self.embedding)
        stack = 1
        self.LSTM1 = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=stack, bidirectional=True) 
        self.LSTM2 = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=stack, bidirectional=True) 
        self.LSTM3 = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=stack, bidirectional=True) 

        self.init_h = Variable(torch.zeros(stack*2, config.batch, config.hidden_size).cuda(device=config.gpu))
        self.init_c = Variable(torch.zeros(stack*2, config.batch, config.hidden_size).cuda(device=config.gpu))
        self.max_pool = nn.MaxPool1d(config.max_seq) 

    def __max_pooling(self, out):
        temp_out = out.permute(1, 2, 0) # [seq, batch, dim] -> [batch, dim, seq]
        output = self.max_pool(temp_out) # [batch, dim, seq = 1]
        u = output.squeeze(2) # [batch, dim, seq = 1] -> [batch, dim]
        return u

    def forward(self, sentence):
        look_up = self.embedding(sentence) # look_up = [seq, batch, dim] # dim = embedding

        out1, (h1, c1) = self.LSTM1(look_up, (self.init_h, self.init_c)) # out = [seq, batch, dim] # dim = hidden
        u1 = self.__max_pooling(out1) # u = [batch, dim * 2] # bi_direct = 2

        out2, (h2, c2) = self.LSTM2(look_up, (h1, c1))
        u2 = self.__max_pooling(out2)

        out3, (h3, c3) = self.LSTM3(look_up, (h2, c2))
        u3 = self.__max_pooling(out3)

        u = torch.cat((u1, u2, u3), 1) # [batch, dim * 2 * 3]
        return u


class Classifier(nn.Module):
    def __init__(self, vocab_list):
        super(Classifier, self).__init__()
        self.sent_enc_p = Sentence_Encoder(vocab_list) 
        self.sent_enc_h = Sentence_Encoder(vocab_list) 
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=config.linear_dropout_keep_prob) 
        
        dim_size= int(config.hidden_size) * 2 * 3 * 4
        self.dense1 = nn.Linear(dim_size, config.linear_hidden_size)
        self.dense2 = nn.Linear(config.linear_hidden_size, config.linear_hidden_size)
        self.dense3 = nn.Linear(config.linear_hidden_size, config.output_class) 

    def forward(self, premise, hypothesis):
        prem = self.sent_enc_p(premise) # [batch, dim * 3]
        hypo = self.sent_enc_p(hypothesis)
        similarity = torch.cat([prem, hypo, torch.abs(prem-hypo), prem*hypo], 1) 
        # print(similarity.shape) # [32, 768] # [batch, dim] (32 * 2 * 3 * 4 = 768)
        dim_size = similarity.shape[1]
        
        fc = self.dense1(similarity)
        fc = self.leaky_relu(fc)
        fc = self.dropout(fc)
 
        fc = self.dense2(fc)
        fc = self.leaky_relu(fc)
        fc = self.dropout(fc)

        out = self.dense3(fc)
        
        return out
        
        

