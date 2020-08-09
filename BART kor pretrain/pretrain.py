import transformer
import create_spm.spm as spm
from torch.utils.data import DataLoader
import torch
# import torch.nn as nn
from torch import nn
import os

from preprocessing import custom_dataset

batch_size = 16
eos_token_id = 2
model_path = "model.pth"

if __name__ == "__main__":
    # spm
    ko_spm, en_spm = spm.get_spm()
    ko_vocab_size = ko_spm.get_vocab_size()

    # train dataset
    ko_paths = ['./data/korean-english-park.dev.ko', './data/korean-english-park.train.ko']
    train_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_paths, mask_token_index=4)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # test dataset
    ko_test_paths = ['./data/korean-english-park.test.ko']
    test_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_test_paths, mask_token_index=4)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

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


    # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    if os.path.isfile(model_path):
        print("pretrain model exist")
        checkpoint = torch.load("model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("previous epoch : ", epoch, " loss : ", loss)
        model.train()

    
    acc_test_list = []
    step = 0
    for epoch in range(100):
        for n, (ko_enc, ko_dec, ko_tar) in enumerate(train_data_loader):
            
            # train
            optimizer.zero_grad()
            logit = model(ko_enc.cuda(), ko_dec.cuda())

            ko_tar = ko_tar.view(-1) # batch flat

            loss = loss_function(logit, ko_tar.cuda())
            loss.backward()
            optimizer.step()

            step += 1
            if step % 200 == 0:
                print("epoch : ", epoch, " step : ", step, " loss : ", loss.item())

            # test
            if step % 3000 == 0:
                for n, (ko_enc, ko_dec, ko_tar) in enumerate(test_data_loader):
                    
                    out = model(ko_enc.cuda(), ko_dec.cuda())
                    _, pred = torch.max(out.data, 1)

                    ko_tar = ko_tar.view(-1)

                    if n == 0:
                        # EOS 이후 토큰 부터 제거  
                        pred_0 = pred.view(batch_size, -1)[0].tolist()
                        new_pred_0 = []
                        for token_index in pred_0:
                            if token_index == eos_token_id: 
                                break
                            new_pred_0.append(token_index)

                        ko_tar_0 = ko_tar.view(batch_size, -1)[0].tolist()
                        new_ko_tar_0 = []
                        for token_index in ko_tar_0:
                            if token_index == eos_token_id: 
                                break
                            new_ko_tar_0.append(token_index)

                        print(ko_spm.decode(new_pred_0))
                        print(ko_spm.decode(new_ko_tar_0))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, model_path)
                print("model save")

