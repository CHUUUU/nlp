import config.config as c
from common_module import glu_custom
import torch.nn as nn
import torch
from torch.autograd import Variable

class conditioning_augmentation_network(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(conditioning_augmentation_network, self).__init__()
        self.fc = nn.Linear(c.hidden_dim, c.cond_aug_norm_dim * 4, bias=True)
        self.glu = glu_custom()

    def encode(self, text_embedding):
        fc = self.glu(self.fc(text_embedding.cuda()))  # fc = [batch, 400]
        mu = fc[:, :c.cond_aug_norm_dim]  # 0-100
        log_var = fc[:, c.cond_aug_norm_dim:]  # 100-200
        return mu, log_var

    # ####################
    # fc: Linear(in_features=256, out_features=400, bias=True)
    # custom_glu, 400 -> 200 , 200 dim drop
    # x: torch.Size([10, 200])
    # mu: torch.Size([10, 100])
    # log_var: torch.Size([10, 100])
    # ####################

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # if c.use_CUDA:
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        text_cond_aug_sample = eps.mul(std).add_(mu)  # sampling by epsilon
        return text_cond_aug_sample

    def forward(self, text_embedding):
        mu, log_var = self.encode(text_embedding)
        text_cond_aug_sample = self.reparametrize(mu, log_var)
        return text_cond_aug_sample, mu, log_var
