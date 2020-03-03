import os
import random
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn

from text_img_pair_dataset import text_img_pair_dataset
import config.config as c
import util as u

from load_pretrain.pretrain_model import LSTM_ENCODER, CNN_ENCODER
from discriminator import D_NET64, D_NET128, D_NET256
from generator import G_NET
from copy import deepcopy

from loss.discriminator_loss import discriminator_loss
from loss.generator_loss import generator_loss
from loss.KL_loss import KL_loss

if __name__ == "__main__":
    seed = 100
    random.seed(seed)
    np.random.seed(seed)

    pair_dataset = text_img_pair_dataset()
    data_loader = torch.utils.data.DataLoader(pair_dataset, batch_size=c.batch_size, drop_last=True, shuffle=True)
    batch_num = len(data_loader)

    text_encoder = LSTM_ENCODER(vocab_size=pair_dataset.vocab_size)
    img_encoder = CNN_ENCODER()
    text_encoder.load_state_dict(torch.load(c.saved_text_model_path))
    img_encoder.load_state_dict(torch.load(c.saved_img_model_path))
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in img_encoder.parameters():
        p.requires_grad = False
    # text_encoder.eval()
    # img_encoder.eval()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    netG = nn.DataParallel(G_NET()).cuda()
    # avg_G = deepcopy(list(p.data for p in netG.parameters()))  # .to("cuda")
    # print(avg_G)
    netsD = []
    netsD.append(nn.DataParallel(D_NET64()).cuda())
    netsD.append(nn.DataParallel(D_NET128()).cuda())
    netsD.append(nn.DataParallel(D_NET256()).cuda())

    optimizerD_list = [optim.Adam(netsD[i].parameters(), lr=0.0002, betas=(0.5, 0.999)) for i in range(len(netsD))]
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_labels = Variable(torch.FloatTensor(c.batch_size).fill_(1)).cuda()
    fake_labels = Variable(torch.FloatTensor(c.batch_size).fill_(0)).cuda()

    noise = Variable(torch.FloatTensor(c.batch_size, c.z_dim)).cuda()
    fixed_noise = Variable(torch.FloatTensor(c.batch_size, c.z_dim).normal_(0, 1)).cuda()

    G_loss_log_list = []
    KL_loss_log_list = []
    D_loss_log_list = [[], [], []]
    for epoch in range(c.epoch):
        data_iter = iter(data_loader)

        step = 0
        while step < batch_num:
            text, index_text, img, cls_id, key = data_iter.next()  # batch
            step += 1
            print("epoch : ", epoch)
            print("step : " + str(step) + "/" + str(batch_num))
            img_256 = img[-1]
            index_text = torch.stack(index_text)
            index_text = torch.transpose(index_text, 1, 0)

            # index_text = index_text.cuda()
            # img = img.cuda()
            cls_id = cls_id.cuda()

            regions_feature, global_feature = img_encoder(img_256)
            words_feature, sent_feature = text_encoder(index_text)
            words_feature, sent_feature = words_feature.detach(), sent_feature.detach()

            regions_feature = regions_feature.cuda()
            global_feature = global_feature.cuda()
            words_feature = words_feature.cuda()
            sent_feature = sent_feature.cuda()

            mask1 = (index_text == pair_dataset.word_2_index[c.EOS_TOKEN]).cuda()  # eos = 23
            mask2 = (index_text == pair_dataset.word_2_index[c.PAD_TOKEN]).cuda()  # pad = 1
            mask = mask1 + mask2

            #######################################################
            # (1) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)  # z sampling
            fake_imgs, att_maps, mu, log_var = netG(noise, sent_feature, words_feature, mask)

            #######################################################
            # (2) Update D network
            ######################################################
            D_loss_total = 0
            D_logs = ''
            for i in range(len(netsD)):
                netsD[i].zero_grad()
                D_loss = discriminator_loss(netsD[i], img[i].cuda(), fake_imgs[i], sent_feature, real_labels, fake_labels)

                D_loss_log_list[i].append(D_loss.data)
                print('D_loss%d: %.2f ' % (i, D_loss.data))

                D_loss.backward()
                optimizerD_list[i].step()
                D_loss_total += D_loss

            #######################################################
            # (3) Update G network: maximize log(D(G(z)))
            ######################################################
            netG.zero_grad()
            G_total_loss, G_logs = generator_loss(netsD, img_encoder, fake_imgs, real_labels, words_feature, sent_feature, cls_id)
            kl_loss = KL_loss(mu, log_var)
            print('G_total_loss: %.2f ' % G_total_loss.data)
            print('kl_loss: %.2f ' % kl_loss.data)

            G_total_loss += kl_loss
            G_loss_log_list.append(G_total_loss.data)  # word + sent
            KL_loss_log_list.append(kl_loss.data)

            G_total_loss.backward()
            optimizerG.step()

            #######################################################
            # (3) show
            ######################################################
            # for p, avg_p in zip(netG.parameters(), avg_G):
            #     avg_p.mul_(0.999).add_(0.001, p.data)

        # every epoch
        u.save_img_results(netG, fixed_noise, sent_feature,
                              words_feature, mask, img_encoder,
                              index_text, epoch, pair_dataset.index_2_word, use_CUDA=False)

        if epoch % 30 == 0 or epoch == (c.epoch-1):  # and epoch != 0:
            torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth' % (c.save_dir_model, epoch))
            # torch.save(netG.state_dict(),'%s/netG_avg_epoch_%d.pth' % (c.save_dir_model, epoch))
            for i in range(len(netsD)):
                torch.save(netsD[i].state_dict(), '%s/netD%d_epoch_%d.pth' % (c.save_dir_model, i, epoch))
            print("epoch : ", epoch)
            print("save model to ", c.save_dir_model)

    u.text_save(c.save_dir_model + 'G_loss_log.txt', G_loss_log_list)
    u.text_save(c.save_dir_model + 'kl_loss_log.txt', KL_loss_log_list)
    u.text_save(c.save_dir_model + 'D1_loss_log.txt', D_loss_log_list[0])
    u.text_save(c.save_dir_model + 'D2_loss_log.txt', D_loss_log_list[1])
    u.text_save(c.save_dir_model + 'D3_loss_log.txt', D_loss_log_list[2])