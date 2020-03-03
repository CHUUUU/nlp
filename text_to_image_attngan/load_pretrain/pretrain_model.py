import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import torch.utils.model_zoo as model_zoo
import config.config as c
import torch.nn.functional as F

class LSTM_ENCODER(nn.Module):
    def __init__(self, vocab_size):
        super(LSTM_ENCODER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, c.embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, -0.1)
        self.drop = nn.Dropout(c.drop_prob)
        self.bi_direction = 2
        self.lstm = nn.LSTM(input_size=c.embedding_dim,
                           hidden_size=c.lstm_hidden_dim,
                           num_layers=c.lstm_stack_layers,
                           batch_first=True,
                           dropout=c.drop_prob,
                           bidirectional=True)

    def forward(self, index_text):
        # index_text = [batch, seq_len]
        emd = self.embedding(index_text)
        lookup = self.drop(emd)

        output, (h_n, c_n) = self.lstm(lookup)
        # words = output, sentence = h_n
        # output = (batch, seq_len, hidden_size * num_directions)
        # h_n = (batch, num_directions * n_layer, hidden_size)

        words_feature = output.transpose(1, 2)
        sent_feature = h_n.transpose(0, 1).contiguous()
        sent_feature = sent_feature.view(-1, c.lstm_hidden_dim * self.bi_direction)
        return words_feature, sent_feature


class CNN_ENCODER(nn.Module):
    def __init__(self):
        super(CNN_ENCODER, self).__init__()

        # pre-train model load
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        # fine-tuning
        self.conv_1x1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(2048, 256)
        self.conv_1x1.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)  # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)  # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 35 x 35 x 192
        x = self.Mixed_5b(x)  # 35 x 35 x 256
        x = self.Mixed_5c(x)  # 35 x 35 x 288
        x = self.Mixed_5d(x)  # 35 x 35 x 288
        x = self.Mixed_6a(x)  # 17 x 17 x 768
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        # image region features
        regions_feature = x  # 17 x 17 x 768
        regions_feature = self.conv_1x1(regions_feature)

        x = self.Mixed_7a(x)  # 8 x 8 x 1280
        x = self.Mixed_7b(x)  # 8 x 8 x 2048
        x = self.Mixed_7c(x)  # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)  # 1 x 1 x 2048
        x = x.view(x.size(0), -1)  # 2048

        # global image features
        global_feature = self.linear(x)  # 512

        return regions_feature, global_feature
