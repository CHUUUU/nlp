import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import config.config as c
from loss.word_loss import words_loss
from loss.sentence_loss import sent_loss


def discriminator_loss(netD, real_imgs, fake_imgs, sent_emb, real_labels, fake_labels):
    cond_real_logits, uncond_real_logits = netD(real_imgs, sent_emb)
    cond_fake_logits, uncond_fake_logits = netD(fake_imgs.detach(), sent_emb)

    cond_real_D_loss = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_D_loss = nn.BCELoss()(cond_fake_logits, fake_labels)

    cond_wrong_logits, _ = netD(real_imgs[:(c.batch_size - 1)], sent_emb[1:c.batch_size])
    cond_wrong_D_loss = nn.BCELoss()(cond_wrong_logits, fake_labels[1:c.batch_size])

    if c.use_uncondition_loss:
        real_D_loss = nn.BCELoss()(uncond_real_logits, real_labels)
        fake_D_loss = nn.BCELoss()(uncond_fake_logits, fake_labels)
        total_D_loss = ((real_D_loss + cond_real_D_loss) / 2. +
                (fake_D_loss + cond_fake_D_loss + cond_wrong_D_loss) / 3.)
    else:
        total_D_loss = cond_real_D_loss + (cond_fake_D_loss + cond_wrong_D_loss) / 2.
    return total_D_loss



