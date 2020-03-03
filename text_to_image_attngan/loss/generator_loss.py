import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import config.config as c
from loss.word_loss import words_loss
from loss.sentence_loss import sent_loss


def generator_loss(netsD, image_encoder, fake_imgs, real_labels, words_embs, sent_emb, cls_ids):
    logs = ''
    G_total_loss = 0
    for i in range(len(netsD)):
        cond_logits, uncond_logits = netsD[i](fake_imgs[i], sent_emb)
        cond_G_loss = nn.BCELoss()(cond_logits, real_labels)
        if c.use_uncondition_loss:
            G_loss = nn.BCELoss()(uncond_logits, real_labels)
            g_loss = G_loss + cond_G_loss
        else:
            g_loss = cond_G_loss
        G_total_loss += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss.data)

        # Ranking loss
        if i == (len(netsD) - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, global_feature = image_encoder(fake_imgs[i].cpu())

            region_features = region_features.cuda()
            global_feature = global_feature.cuda()

            w_loss0, w_loss1, _ = words_loss(region_features, words_embs, cls_ids)
            w_loss = (w_loss0 + w_loss1) * c.LAMBDA

            s_loss0, s_loss1 = sent_loss(global_feature, sent_emb, cls_ids)
            s_loss = (s_loss0 + s_loss1) * c.LAMBDA

            G_total_loss += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.data, s_loss.data)
    return G_total_loss, logs