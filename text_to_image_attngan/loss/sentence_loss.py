import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from attention import region_attention
import config.config as c
import numpy as np
import torch.nn as nn
import util as u


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def words_loss(region_features, words_features, cls_id):
    """
        words_features(query): batch x nef x seq_len
        region_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    nums = c.max_seq - 1
    for i in range(c.batch_size):
        if cls_id is not None:
            mask = (cls_id == cls_id[i])
            mask[i] = 0
            masks.append(mask.reshape((1, -1)).cpu())

        # Get the i-th text description
        batch_1_words_feature = words_features[i, :, : nums].unsqueeze(0).contiguous() # -> 1 x nef x words_num
        batch_1_words_feature = batch_1_words_feature.repeat(c.batch_size, 1, 1)  # -> batch_size x nef x words_num

        """
            word(query): batch x nef x words_num
            region_features : batch x nef x 17 x 17
            weighted_region_features: batch x nef x words_num
            region_attn: batch x words_num x 17 x 17
        """
        weighted_region_features, region_attn = \
            region_attention(batch_1_words_feature, region_features, c.GAMMA1)

        att_maps.append(region_attn[i].unsqueeze(0).contiguous())

        # --> batch_size x words_num x nef
        batch_1_words_feature = batch_1_words_feature.transpose(1, 2).contiguous()
        weighted_region_features = weighted_region_features.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        batch_1_words_feature = batch_1_words_feature.view(c.batch_size * nums, -1)
        weighted_region_features = weighted_region_features.view(c.batch_size * nums, -1)

        # --> batch_size * word_max_seq
        row_sim = cosine_similarity(batch_1_words_feature, weighted_region_features)
        # --> batch_size x words_num
        row_sim = row_sim.view(c.batch_size, nums)

        # Eq. (10)
        row_sim.mul_(c.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    similarities = similarities * c.GAMMA3
    if cls_id is not None:
        masks = np.concatenate(masks, 0)
        # masks = torch.cat(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        masks = masks.cuda()
        similarities.data.masked_fill_(masks, -float('inf'))

    similarities_transpose = similarities.transpose(0, 1)
    if c.train_mode:
        labels = torch.tensor(list(range(c.batch_size)))
        if c.use_CUDA:
            labels = labels.cuda()
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities_transpose, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def sent_loss(global_feature, sent_feature, cls_id):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if cls_id is not None:
        for i in range(c.batch_size):
            mask = (cls_id == cls_id[i])
            mask[i] = 0
            masks.append(mask.reshape((1, -1)).cpu())
        masks = np.concatenate(masks, 0)
        # masks = torch.cat(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        # masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if global_feature.dim() == 2:
        global_feature = global_feature.unsqueeze(0)
        sent_feature = sent_feature.unsqueeze(0)

    # global_feature_norm / sent_feature_norm: seq_len x batch_size x 1
    global_feature_norm = torch.norm(global_feature, 2, dim=2, keepdim=True)
    sent_feature_norm = torch.norm(sent_feature, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(global_feature, sent_feature.transpose(1, 2))
    norm0 = torch.bmm(global_feature_norm, sent_feature_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=1e-8) * c.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if cls_id is not None:
        scores0.data.masked_fill_(masks.cuda(), -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if c.train_mode:
        labels = torch.tensor(list(range(c.batch_size)))
        if c.use_CUDA:
            labels = labels.cuda()
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1
