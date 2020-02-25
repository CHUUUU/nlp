import torch
import torch.nn as nn
import config.config as c

def conv_by_16times(d_dim):
    deconv_img = nn.Sequential(
        nn.Conv2d(3, d_dim, kernel_size=4, stride=2, padding=1, bias=False),  # dim x in_size/2 x in_size/2
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(d_dim, d_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 2*dim x x in_size/4 x in_size/4
        nn.BatchNorm2d(d_dim * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(d_dim * 2, d_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 4*dim x in_size/8 x in_size/8
        nn.BatchNorm2d(d_dim * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(d_dim * 4, d_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),  # 8*dim x in_size/16 x in_size/16
        nn.BatchNorm2d(d_dim * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return deconv_img  # down_sale the spatial size by a factor of 16


def down_block_1(input_dim, output_dim):
    block = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block  # down_sale the spatial size by a factor of 2


def down_block_2(input_dim, output_dim):
    block = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


class D_GET_LOGITS(nn.Module):
    def __init__(self, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        d_dim = c.discrimination_semantic_dim
        self.bcondition = bcondition
        if self.bcondition:
            self.joint_conv = down_block_2(d_dim * 8 + c.hidden_dim, d_dim * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(d_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, img_4x4, sent_emb=None):
        if self.bcondition:
            sent_emb = sent_emb.view(-1, c.hidden_dim, 1, 1)
            sent_emb = sent_emb.repeat(1, 1, 4, 4)
            cat = torch.cat((img_4x4, sent_emb), 1)  # (ngf+egf) x 4 x 4
            output = self.joint_conv(cat)  # ngf x in_size x in_size
        else:
            output = img_4x4
        output = self.outlogits(output)
        return output.view(-1)

# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        d_dim = c.discrimination_semantic_dim
        self.cond_net = D_GET_LOGITS(bcondition=True)
        self.uncond_net = D_GET_LOGITS(bcondition=False)
        self.img_code_s16 = conv_by_16times(d_dim)

    def forward(self, img, sent_emb):
        x = self.img_code_s16(img)  # 4 x 4 x 8df
        cond_logit = self.cond_net(x, sent_emb)
        uncond_logit = self.uncond_net(x)
        return cond_logit, uncond_logit


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        d_dim = c.discrimination_semantic_dim
        self.cond_net = D_GET_LOGITS(bcondition=True)
        self.uncond_net = D_GET_LOGITS(bcondition=False)
        self.img_code_s16 = conv_by_16times(d_dim)
        self.img_code_s32 = down_block_1(d_dim * 8, d_dim * 16)
        self.img_code_s32_1 = down_block_2(d_dim * 16, d_dim * 8)

    def forward(self, img, sent_emb):
        x = self.img_code_s16(img)   # 8 x 8 x 8df
        x = self.img_code_s32(x)   # 4 x 4 x 16df
        x = self.img_code_s32_1(x)  # 4 x 4 x 8df
        cond_logit = self.cond_net(x, sent_emb)
        uncond_logit = self.uncond_net(x)
        return cond_logit, uncond_logit


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        d_dim = c.discrimination_semantic_dim
        self.cond_net = D_GET_LOGITS(bcondition=True)
        self.uncond_net = D_GET_LOGITS(bcondition=False)
        self.conv_16 = conv_by_16times(d_dim)
        self.img_code_s32 = down_block_1(d_dim * 8, d_dim * 16)
        self.img_code_s64 = down_block_1(d_dim * 16, d_dim * 32)
        self.img_code_s64_1 = down_block_2(d_dim * 32, d_dim * 16)
        self.img_code_s64_2 = down_block_2(d_dim * 16, d_dim * 8)

    def forward(self, img, sent_emb):
        x = self.conv_16(img)
        x = self.img_code_s32(x)
        x = self.img_code_s64(x)
        x = self.img_code_s64_1(x)
        x = self.img_code_s64_2(x)
        cond_logit = self.cond_net(x, sent_emb)
        uncond_logit = self.uncond_net(x)
        return cond_logit, uncond_logit
