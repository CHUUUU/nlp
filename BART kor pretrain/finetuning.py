import transformer
import create_spm.spm as spm
from torch.utils.data import DataLoader
import torch
# import torch.nn as nn
from torch import nn
import os

from sentiment_classification import binary_classification, nsmc_dataset

batch_size = 32
model_path = "model.pth"
finetuning_model_path = "model_cls.pth"

if __name__ == "__main__":
    # spm
    ko_spm, en_spm = spm.get_spm()
    ko_vocab_size = ko_spm.get_vocab_size()

    # train dataset
    nsmc_path = ['./finetuning_data/nsmc/ratings_train.txt']
    train_data = nsmc_dataset(ko_spm=ko_spm, nsmc_path=nsmc_path)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # test dataset
    nsmc_test_path = ['./finetuning_data/nsmc/ratings_test.txt']
    test_data = nsmc_dataset(ko_spm=ko_spm, nsmc_path=nsmc_test_path)
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

    
    # pretrain model load
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model = model.cuda()
    if os.path.isfile(model_path):
        print("model exist")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # print("previous epoch : ", epoch, " loss : ", loss)
        model.train()
    

    # fine-tuning model
    finetuning_model = binary_classification(bart_model=model, freeze_bart=True, vocab_size=ko_vocab_size)
    finetuning_model = torch.nn.DataParallel(finetuning_model, device_ids=[0]).cuda()
    # finetuning_model = finetuning_model.cuda()

    optimizer = torch.optim.Adam(finetuning_model.parameters(), lr=0.00015)
    loss_function = nn.BCELoss() 
    if os.path.isfile(finetuning_model_path):
        print("finetuning_model exist")
        checkpoint = torch.load(finetuning_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("previous epoch : ", epoch, " loss : ", loss)
        model.train()

    # train
    acc_test_list = []
    step = 0
    for epoch in range(100):
        for n, (ko_enc, ko_dec, cls_label, last_token_position) in enumerate(train_data_loader):
            optimizer.zero_grad()
            logit = finetuning_model(ko_enc.cuda(), ko_dec.cuda(), last_token_position.cuda(), batch_size)

            cls_label = cls_label.type('torch.FloatTensor')
 
            # batch flat
            logit = logit.squeeze(-1)

            loss = loss_function(logit, cls_label.cuda()) # [32], [32]
            loss.backward() ### error
            optimizer.step()

            step += 1
            if step % 200 == 0:
                print("epoch : ", epoch, " step : ", step, " loss : ", loss.item())

            if step % 1000 == 0:
                for n, (ko_enc, ko_dec, ko_tar) in enumerate(test_data_loader):
                    out = model(ko_enc.cuda(), ko_dec.cuda())
                    cls_out = finetuning_model(ko_enc.cuda(), ko_dec.cuda())

                    _, pred = torch.max(out.data, 1)

                    ko_tar = ko_tar.view(-1)

                    if n == 0:
                        # BOS 1 이후 토큰 부터 제거  
                        pred_0 = pred.view(batch_size, -1)[0].tolist()
                        new_pred_0 = []
                        for token_index in pred_0:
                            if token_index == 1:  # BOS Token index
                                break
                            new_pred_0.append(token_index)

                        ko_tar_0 = ko_tar.view(batch_size, -1)[0].tolist()
                        new_ko_tar_0 = []
                        for token_index in ko_tar_0:
                            if token_index == 1:  # BOS Token index
                                break
                            new_ko_tar_0.append(token_index)

                        print(ko_spm.decode(new_pred_0))
                        print(ko_spm.decode(new_ko_tar_0))
                        print(cls_out[0])

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, finetuning_model_path)
                print("model save")

