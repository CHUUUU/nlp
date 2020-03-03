import os
import pickle
import json
import torch
import torch.nn as nn
import config.config as c
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import skimage.transform
from loss.word_loss import words_loss


def pickle_load(path):
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        print("load error : ", path)
        assert False


def text_load(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().split('\n')
        return data
    else:
        print("load error : ", path)
        assert False


def text_save(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def json_load(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    else:
        print("load error : ", path)
        assert False


def json_save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=c.batch_size,
                       max_word_num=c.max_seq):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        if i >= len(COLOR_DIC):
            break
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    img_txt = Image.fromarray(convas)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []

    for i in range(c.batch_size):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[str(cap[j])]
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def save_img_results(netG, noise, sent_emb, words_embs, mask, image_encoder, captions,
                     epoch, ixtoword, name='average', use_CUDA=True):
    # Save images
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
    for i in range(len(attention_maps)):
        if len(fake_imgs) > 1:
            img = fake_imgs[i + 1].detach().cpu()
            lr_img = fake_imgs[i].detach().cpu()
        else:
            img = fake_imgs[0].detach().cpu()
            lr_img = None
        attn_maps = attention_maps[i]
        att_sze = attn_maps.size(2)
        img_set, _ = \
            build_super_images(img, captions, ixtoword,
                               attn_maps, att_sze, lr_imgs=lr_img)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_%d_%d.png' % (c.save_dir_test, name, epoch, i)
            im.save(fullpath)

    # for i in range(len(netsD)):
    i = -1
    img = fake_imgs[i].detach().cpu()
    region_features, _ = image_encoder(img)
    att_sze = region_features.size(2)
    _, _, att_maps = words_loss(region_features.detach(),
                                words_embs.detach().cpu(),
                                None, use_CUDA)
    img_set, _ = \
        build_super_images(fake_imgs[i].detach().cpu(),
                           captions, ixtoword, att_maps, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/D_%s_%d.png' % (c.save_dir_test, name, epoch)
        im.save(fullpath)