import transformer
import create_spm.spm as spm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

from preprocessing import custom_dataset

batch_size = 32

if __name__ == "__main__":
    # spm
    ko_spm, en_spm = spm.get_spm()
    ko_vocab_size = ko_spm.get_vocab_size()
    # en_vocab_size = en_spm.get_vocab_size()

    # train dataset
    ko_paths = ['./data/korean-english-park.dev.ko', './data/korean-english-park.train.ko']
    # en_paths = ['./data/korean-english-park.dev.en', './data/korean-english-park.train.en']
    train_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_paths, mask_token_index=4)
    # train_data = custom_dataset(ko_spm=ko_spm, en_spm=en_spm, ko_paths=ko_paths, en_paths=en_paths)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True) # , collate_fn=train_data.collate)

    # test dataset
    ko_test_paths = ['./data/korean-english-park.test.ko']
    # en_test_paths = ['./data/korean-english-park.test.en']
    test_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_test_paths, mask_token_index=4)
    # test_data = custom_dataset(ko_spm=ko_spm, en_spm=en_spm, ko_paths=ko_test_paths, en_paths=en_test_paths)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True) #, collate_fn=test_data.collate)

    # model setting
    model = transformer.Transformer(n_src_vocab=ko_vocab_size,
                                    n_trg_vocab=None,
                                    src_pad_idx=0,
                                    trg_pad_idx=0,
                                    d_word_vec=128,
                                    d_model=128,
                                    d_inner=512,
                                    n_layers=3,
                                    n_head=4,
                                    d_k=32,
                                    d_v=32,
                                    dropout=0.1,
                                    n_position=256,
                                    trg_emb_prj_weight_sharing=True,
                                    emb_src_trg_weight_sharing=True)

    # model = transformer.Transformer(n_src_vocab=ko_vocab_size,
    #                                 n_trg_vocab=en_vocab_size,
    #                                 src_pad_idx=0,
    #                                 trg_pad_idx=0,
    #                                 d_word_vec=128,
    #                                 d_model=128,
    #                                 d_inner=512,
    #                                 n_layers=3,
    #                                 n_head=4,
    #                                 d_k=32,
    #                                 d_v=32,
    #                                 dropout=0.1,
    #                                 n_position=256,
    #                                 trg_emb_prj_weight_sharing=True,
    #                                 emb_src_trg_weight_sharing=False)

    model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile("model.pth"):
        print("model exist")
        model.load_state_dict(torch.load("model.pth"))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    # train
    acc_test_list = []
    step = 0
    for i in range(20):
        for n, (ko_enc, ko_dec, ko_tar) in enumerate(train_data_loader):
            optimizer.zero_grad()
            logit = model(ko_enc.cuda(), ko_dec.cuda())

            # batch flat
            ko_tar = ko_tar.view(-1)

            loss = loss_function(logit, ko_tar.cuda())
            loss.backward()
            optimizer.step()

            step += 1
            if step % 200 == 0:
                print("epoch : ", i, " step : ", step, " loss : ", loss.item())

            if step % 1000 == 0:
                # total = 0
                # correct = 0
                for n, (ko_enc, ko_dec, ko_tar) in enumerate(test_data_loader):
                    out = model(ko_enc.cuda(), ko_dec.cuda())

                    _, pred = torch.max(out.data, 1)

                    # ★ ACC 측정 변경해야함 
                    # total += batch_size*186
                    # correct += (pred == ko_tar.cuda()).sum()
                    ko_tar = ko_tar.view(-1)

                    if n == 0:
                        print(ko_spm.decode(pred.view(batch_size, -1)[0].tolist()))
                        print(ko_spm.decode(ko_tar.view(batch_size, -1)[0].tolist()))

                # acc = 100 * (correct.cpu().numpy() / total)
                # acc_test_list.append(acc)
                torch.save(model.state_dict(), "model.pth")
                print("model save, acc : ", acc)


    print("test acc : ", acc_test_list)

