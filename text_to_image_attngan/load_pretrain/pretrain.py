import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
import numpy as np
import torch
import torch.nn as nn

import util as u
import config.config as c

from PIL import Image
from torch.autograd import Variable

from text_img_pair_dataset import text_img_pair_dataset
from loss.word_loss import words_loss
from loss.sentence_loss import sent_loss
from loss.pretrain_model import LSTM_ENCODER, CNN_ENCODER


# cpu train
if __name__=="__main__":
    pair_dataset = text_img_pair_dataset()
    text_encoder = LSTM_ENCODER(vocab_size=pair_dataset.vocab_size)
    img_encoder = CNN_ENCODER()

    # 훈련
    if not os.path.isfile(c.saved_text_model_path):
        start_time = time.time()
        # optimizer
        para = list(text_encoder.parameters())
        for v in img_encoder.parameters():
            if v.requires_grad:
                para.append(v)
        pair_optimizer = torch.optim.Adam(para, lr=0.0002, betas=(0.5, 0.999))

        # train
        # text_encoder.cuda()
        # img_encoder.cuda()
        text_encoder.train()
        img_encoder.train()

        w0_loss_list = []
        w1_loss_list = []
        s0_loss_list = []
        s1_loss_list = []

        w_loss0_cumulation = 0
        w_loss1_cumulation = 0
        s_loss0_cumulation = 0
        s_loss1_cumulation = 0
        word_total_loss = 0
        sent_total_loss = 0

        for epoch in range(c.epoch):
            data_loader = torch.utils.data.DataLoader(
                pair_dataset, batch_size=c.batch_size, drop_last=True,
                shuffle=True, num_workers=int(0))

            for step, data in enumerate(data_loader):
                text_encoder.zero_grad()
                img_encoder.zero_grad()

                text, index_text, img, cls_id, key = data

                img = img[-1]
                index_text = torch.stack(index_text)
                index_text = torch.transpose(index_text, 1, 0)  # [word_seq, batch_size] -> [batch_size, word_seq]

                # img = Variable(img.cuda())
                # index_text = Variable(index_text.cuda())
                # cls_id = Variable(cls_id.cuda())

                regions_feature, global_feature = img_encoder(img)
                words_feature, sent_feature = text_encoder(index_text)

                # regions_feature = Variable(regions_feature.cuda())
                # global_feature = Variable(global_feature.cuda())
                # words_feature = Variable(words_feature.cuda())
                # sent_feature = Variable(sent_feature.cuda())

                w_loss0, w_loss1, attn_maps = words_loss(regions_feature, words_feature, cls_id)
                w_loss0_cumulation += w_loss0.data
                w_loss1_cumulation += w_loss1.data

                s_loss0, s_loss1 = sent_loss(global_feature, sent_feature, cls_id)
                s_loss0_cumulation += s_loss0.data
                s_loss1_cumulation += s_loss1.data

                # loss = (w_loss0 + w_loss1 + s_loss0 + s_loss1).data
                loss = w_loss0 + w_loss1 + s_loss0 + s_loss1
                loss.backward()

                # `clip_grad_norm` helps prevent
                # the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(text_encoder.parameters(), 0.25)

                pair_optimizer.step()

                w0_loss_list.append(w_loss0.data)
                w1_loss_list.append(w_loss1.data)
                s0_loss_list.append(s_loss0.data)
                s1_loss_list.append(s_loss1.data)

                if step % 100 == 0:
                    w_loss0_cumulation /= 100
                    w_loss1_cumulation /= 100
                    s_loss0_cumulation /= 100
                    s_loss1_cumulation /= 100
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                          's_loss {:5.2f} {:5.2f} | '
                          'w_loss {:5.2f} {:5.2f}'
                          .format(epoch, step, len(data_loader),
                                  (time.time() - start_time),
                                  s_loss0_cumulation, s_loss1_cumulation,
                                  w_loss0_cumulation, w_loss1_cumulation))
                    w_loss0_cumulation = 0
                    w_loss1_cumulation = 0
                    s_loss0_cumulation = 0
                    s_loss1_cumulation = 0

                    img_set, _ = u.build_super_images(img.cpu(),
                                                      index_text.view(c.batch_size, -1),
                                                      pair_dataset.index_2_word,
                                                      attn_maps,
                                                      regions_feature.size(2))
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        att_path = c.pre_train_save_dir + 'epoch_%d_step_%d_pretrain_attention_maps.png' % (epoch, step)
                        im.save(att_path)


        torch.save(text_encoder.state_dict(), c.saved_text_model_path)
        torch.save(img_encoder.state_dict(), c.saved_img_model_path)

        u.text_save(c.pre_train_save_dir + 'w0_loss_list.txt', w0_loss_list)
        u.text_save(c.pre_train_save_dir + 'w1_loss_list.txt', w1_loss_list)
        u.text_save(c.pre_train_save_dir + 's0_loss_list.txt', s0_loss_list)
        u.text_save(c.pre_train_save_dir + 's1_loss_list.txt', s1_loss_list)

    text_encoder.load_state_dict(torch.load(c.saved_text_model_path))
    img_encoder.load_state_dict(torch.load(c.saved_img_model_path))







