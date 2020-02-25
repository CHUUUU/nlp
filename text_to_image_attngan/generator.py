import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config.config as c
import torch
import torch.nn as nn
from common_module import glu_custom
from conditioning_augmentation_network import conditioning_augmentation_network as ca_net
from attention import global_attention


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_num * 2),
            glu_custom(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# Upsale the spatial size by a factor of 2
def upsample_custom(input_dim, output_dim):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(input_dim, output_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_dim * 2),
        glu_custom())
    return block


class INIT_STAGE_G(nn.Module):
    def __init__(self):
        super(INIT_STAGE_G, self).__init__()
        self.g_dim = c.generation_semantic_dim * 16  # 2048 = 128 * 16
        input_dim = c.z_dim + c.cond_aug_norm_dim  # 200 = 100 + 100

        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.g_dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.g_dim * 4 * 4 * 2),
            glu_custom())

        # 128 * 16 * 4 * 4 * 2 = 65,536‬
        # 65,536‬ / (64 * 2) = 16
        # 16 * 64 * 64

        self.upsample1 = upsample_custom(self.g_dim, self.g_dim // 2).cuda()
        self.upsample2 = upsample_custom(self.g_dim // 2, self.g_dim // 4).cuda()
        self.upsample3 = upsample_custom(self.g_dim // 4, self.g_dim // 8).cuda()
        self.upsample4 = upsample_custom(self.g_dim // 8, self.g_dim // 16).cuda()


    def forward(self, z, text_cond_aug_sample):
        cat_c_z = torch.cat((text_cond_aug_sample, z), 1)
        fc = self.fc(cat_c_z)  # 65536
        fc = fc.view(-1, self.g_dim, 4, 4)  # 4096 x 4 x 4
        up = self.upsample1(fc)  # 1024 x 8 x 8
        up = self.upsample2(up)  # 256 x 16 x 16
        up = self.upsample3(up)  # 64 x 32 x 32
        up = self.upsample4(up)  # 16 x 64 x 64

        return up  # batch x 16 x 64 x 64


class NEXT_STAGE_G(nn.Module):
    def __init__(self):
        super(NEXT_STAGE_G, self).__init__()
        self.global_attn = global_attention()

        layers = []
        for i in range(c.num_residual_layer):
            layers.append(ResBlock(c.generation_semantic_dim * 2))
        self.residual = nn.Sequential(*layers)
        self.upsample = upsample_custom(c.generation_semantic_dim * 2, c.generation_semantic_dim)

    def forward(self, previous_generation, word_embs, mask):
        # previous_generation : batch x dim x h x w
        # word_embs : batch x dim x seq_len
        # weight_global_feature : batch x dim x (h x w)

        self.global_attn.applyMask(mask)
        weight_global_feature, global_attn = self.global_attn(previous_generation, word_embs)
        cat = torch.cat((previous_generation, weight_global_feature), 1)
        res = self.residual(cat)
        up = self.upsample(res)  # dim/4 x h*2 x w*2

        return up, global_attn  # global_attn : [batch, seq_len, (h x w)]


class generate_image(nn.Module):
    def __init__(self):
        super(generate_image, self).__init__()
        self.img = nn.Sequential(
            nn.Conv2d(c.generation_semantic_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.ca_net = ca_net().cuda()

        # 64 x 64
        self.G_first = INIT_STAGE_G()
        self.gen_img1 = generate_image()

        # 128 x 128
        self.G_next1 = NEXT_STAGE_G()
        self.gen_img2 = generate_image()

        # 256 x 256
        self.G_next2 = NEXT_STAGE_G()
        self.gen_img3 = generate_image()

    def forward(self, z, sent_emb, word_embs, mask):
        """
            :param z: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
        """
        fake_imgs = []
        att_maps = []
        text_cond_aug_sample, mu, log_var = self.ca_net(sent_emb)

        # 64 x 64
        g1 = self.G_first(z, text_cond_aug_sample)
        fake_img1 = self.gen_img1(g1)
        fake_imgs.append(fake_img1)

        # 128 x 128
        g2, att1 = self.G_next1(g1, word_embs, mask)
        fake_img2 = self.gen_img2(g2)
        fake_imgs.append(fake_img2)
        if att1 is not None:
            att_maps.append(att1)

        # 256 x 256
        g3, att2 = self.G_next2(g2, word_embs, mask)
        fake_img3 = self.gen_img3(g3)
        fake_imgs.append(fake_img3)
        if att2 is not None:
            att_maps.append(att2)

        return fake_imgs, att_maps, mu, log_var