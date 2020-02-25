import torch
import torch.nn as nn
import config.config as c


def region_attention(word_features, region_features, gamma1):
    """
    word_features: batch x ndf x word_seq_length
    region_features: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """

    batch_size, word_seq_length = word_features.size(0), word_features.size(2)
    ih, iw = region_features.size(2), region_features.size(3)
    region_seq_length = ih * iw

    # --> batch x region_seq_length x ndf
    region_features = region_features.view(batch_size, -1, region_seq_length)
    region_featuresT = torch.transpose(region_features, 1, 2).contiguous()

    # Get attention (# Eq. (7) in AttnGAN paper)
    # (batch x region_seq_length x ndf)(batch x ndf x word_seq_length)
    attn = torch.bmm(region_featuresT, word_features)  # --> batch x region_seq_length x word_seq_length
    attn = attn.view(-1, word_seq_length)  # --> batch*region_seq_length x word_seq_length
    attn = nn.Softmax()(attn)  # Eq. (8), region_1 = w1 + w2 + w3 + ..., compute prob by each row

    attn = attn.view(batch_size, region_seq_length, word_seq_length)  # --> batch x region_seq_length x word_seq_length
    attn = torch.transpose(attn, 1, 2).contiguous()  # --> batch * word_seq_length x region_seq_length
    attn = attn.view(batch_size * word_seq_length, region_seq_length)

    attn = attn * gamma1  # Eq. (9)
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, word_seq_length, region_seq_length)
    # --> batch x region_seq_length x word_seq_length
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x region_seq_length)(batch x region_seq_length x word_seq_length)
    # --> batch x ndf x word_seq_length
    weighted_region_features = torch.bmm(region_features, attnT)

    region_attention = attn.view(batch_size, -1, ih, iw)

    return weighted_region_features, region_attention


class global_attention(nn.Module):
    def __init__(self):
        super(global_attention, self).__init__()
        self.conv_1x1 = nn.Conv2d(c.hidden_dim, c.generation_semantic_dim, kernel_size=1, stride=1, padding=0, bias=False).cuda()
        self.softmax = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, previous_generation_img, word_embs):
        """
            previous_generation_img: batch x idf x ih x iw (word_featuresL=ihxiw)
            word_embs: batch x cdf x sourceL
        """
        ih, iw = previous_generation_img.size(2), previous_generation_img.size(3)
        batch_size, word_seq_len = word_embs.size(0), word_embs.size(2)

        img_target = previous_generation_img.view(batch_size, -1, ih * iw)  # batch x g_semantic_dim x (ih * iw)
        img_targetT = torch.transpose(img_target, 1, 2).contiguous()  # batch x (ih * iw) x g_semantic_dim

        word_sourceT = word_embs.unsqueeze(3)  # batch x c.hidden x word_seq_len --> batch x c.hidden x word_seq_len x 1
        word_sourceT = self.conv_1x1(word_sourceT).squeeze(3)  # batch x g_semantic_dim x word_seq_len

        # Get attention
        # (batch x ih * iw x g_semantic_dim)(batch x g_semantic_dim x word_seq_len)
        attn = torch.bmm(img_targetT, word_sourceT)  # batch x (ih * iw) x word_seq_len
        attn = attn.view(batch_size * ih * iw, word_seq_len)  # (batch * ih * iw) x word_seq_len
        if self.mask is not None:
            mask = self.mask.repeat(ih * iw, 1)  # batch_size x word_seq_len --> (batch_size * ih * iw) x word_seq_len
            attn.data.masked_fill_(mask.data, -float('inf'))  # eos, pad -> -inf
        attn = self.softmax(attn)  # Eq. (2)
        attn = attn.view(batch_size, ih * iw, word_seq_len)  # batch x ih * iw x word_seq_len
        attn = torch.transpose(attn, 1, 2).contiguous()  # batch x word_seq_len x ih * iw

        # (batch x g_semantic_dim x word_seq_len)(batch x word_seq_len x ih * iw)
        weighted_global_features = torch.bmm(word_sourceT, attn)  # batch x g_semantic_dim x (ih * iw)
        weighted_global_features = weighted_global_features.view(batch_size, -1, ih, iw)  # batch x g_semantic_dim x ih x iw
        global_attention = attn.view(batch_size, -1, ih, iw)

        return weighted_global_features, global_attention
