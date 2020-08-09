import transformer
import create_spm.spm as spm
from torch.utils.data import DataLoader
import torch
from torch import nn
import os
import numpy as np

from sentiment_classification import binary_classification, nsmc_dataset

batch_size = 32
eos_token_id = 2
model_path = "model.pth"
finetuning_model_path = "model_cls.pth"

if __name__ == "__main__":
    # spm
    ko_spm, en_spm = spm.get_spm()
    ko_vocab_size = ko_spm.get_vocab_size()

    # train dataset
    nsmc_path = ['./data_finetuning/nsmc/ratings_train.txt']
    train_data = nsmc_dataset(ko_spm=ko_spm, nsmc_path=nsmc_path)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # test dataset
    nsmc_test_path = ['./data_finetuning/nsmc/ratings_test.txt']
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

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    finetuning_model = binary_classification(bart_model=model, freeze_bart=False, vocab_size=ko_vocab_size).cuda()
    optimizer = torch.optim.Adam([{"params" : finetuning_model.bart.parameters()},
                                {"params" : finetuning_model.cls_layer.parameters()}], lr=0.0015)
    loss_function = nn.CrossEntropyLoss()
    
    # model load
    if os.path.isfile(finetuning_model_path):
        print("finetuning model exist")
        checkpoint = torch.load(finetuning_model_path)
        model.load_state_dict(checkpoint['model_state_dict_pretrain'])
        finetuning_model.load_state_dict(checkpoint['model_state_dict_finetuning'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        print("previous epoch : ", epoch, " loss : ", loss, "acc : ", acc)
        model.train()
        finetuning_model.train()
    elif os.path.isfile(model_path):
        print("pretrain model exist")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
    else:
        print("no model")

   
    acc_test_list = []
    step = 0
    for epoch in range(100):
        for n, (ko_enc, ko_dec, cls_label, last_token_position) in enumerate(train_data_loader):       
            
            # train
            optimizer.zero_grad()
            logit = finetuning_model(ko_enc.cuda(), ko_dec.cuda(), last_token_position.cuda(), batch_size)

            loss = loss_function(logit, cls_label.cuda()) # [32,2], [32]
            loss.backward() 
            optimizer.step()

            step += 1
            if step % 200 == 0:
                print("epoch : ", epoch, " step : ", step, " loss : ", loss.item())

            # test
            total = 0
            correct = 0
            if step % 1000 == 0:
                for n, (ko_enc, ko_dec, cls_label, last_token_position) in enumerate(test_data_loader):
                    
                    cls_out = finetuning_model(ko_enc.cuda(), ko_dec.cuda(), last_token_position.cuda(), batch_size)
                    
                    # check a result with test batch[0]
                    if n == 0:
                        # original input
                        ko_enc = ko_enc.view(batch_size, -1)[0].tolist()
                        new_ko_tar_0 = []
                        for token_index in ko_enc:
                            if token_index == eos_token_id: # <EOS> 이후 모든 토큰 제거  
                                break
                            new_ko_tar_0.append(token_index)
                        print(ko_spm.decode(new_ko_tar_0))

                        # language_model inference
                        # out = model(ko_enc.cuda(), ko_dec.cuda())
                        # _, pred = torch.max(out.data, 1)
                        # pred_0 = pred.view(batch_size, -1)[0].tolist()
                        # new_pred_0 = []
                        # # print(pred_0)
                        # for token_index in pred_0:
                        #     if token_index == eos_token_id:  # eos 부터 제거
                        #         break
                        #     new_pred_0.append(token_index)
                        # print(ko_spm.decode(new_pred_0))

                        # cls inference
                        _, pred = torch.max(cls_out.data, 1)
                        print("cls inference : ", pred[0].data)
                
                    # ACC
                    _, pred = torch.max(cls_out.data, 1)
                    total += len(cls_label) # batch size
                    correct += np.array((pred.data.cpu() == cls_label)).sum() 
                acc = 100 * (correct/total)
                acc_test_list.append(acc)
                print("acc : ", acc)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict_pretrain': finetuning_model.bart.state_dict(),
                    'model_state_dict_finetuning': finetuning_model.cls_layer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'acc': acc
                }, finetuning_model_path)
                print("model save")

