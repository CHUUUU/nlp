import torch
import torch.nn as nn
import preprocessing.custom_dataset as custom_dataset
import model as model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config as config
import numpy as np
import matplotlib.pyplot as plt
import preprocessing.vocab as v
import time

def eval(loader):
    total = 0
    correct = 0
    for n, (label, sent) in enumerate(loader):
        label = Variable(label.to(device))
        sent = Variable(torch.stack(sent).to(device))
        out =  model(sent)
        _, pred = torch.max(out.data, 1)
        total += label.size(0) # batch size
        correct += (pred == label).sum() 
    acc = 100 * (correct.cpu().numpy()/total)
    return acc

if __name__ == "__main__":
    path_csv = config.path_csv

    # 데이터 처리
    start = time.time()
    vocab = v.create_vocab(path_csv=path_csv)
    word_to_index = vocab.get_data()
    print("time vocab load : ", time.time() - start)

    start = time.time()
    dataset = custom_dataset.Custom_dataset(word_to_index, path_csv=path_csv)
    train_data = dataset.get_data()
    print("데이터 준비 완료")
    print("time data load : ", time.time() - start)

    print(len(train_data))

    train_loader = DataLoader(train_data,
                            batch_size=config.batch,
                            shuffle=True,
                            # num_workers=config.cpu_processor,
                            drop_last=True)


    test_dataset = custom_dataset.Custom_dataset(word_to_index, path_csv="train_data.csv")
    test_data = test_dataset.get_data()
    test_loader = DataLoader(test_data,
                        batch_size=config.batch,
                        shuffle=False,
                        num_workers=config.cpu_processor,
                        drop_last=True)

    # 모델 설정
    device = 'cpu' #torch.device(config.gpu if torch.cuda.is_available() else 'cpu')
    model = model.classifier(len(word_to_index))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 훈련
    step_list = []
    loss_list = []
    acc_test_list = []
    acc_dev_list = []
    step = 0
    for i in range(config.epoch):
        print("epoch = ", i)
        start = time.time()
        for n, (label, sent) in enumerate(train_loader):

            optimizer.zero_grad() # 초기화
            
            sent = Variable(torch.stack(sent).to(device))
            label = Variable(label.to(device))
            
            logit = model(sent)
            
            print(label)
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            step += 1
            if n % 20 == 0:
                print("epoch : ", i, " step : ", n , " loss : ", loss.item())
                step_list.append(step)
                loss_list.append(loss)
                acc_test = eval(test_loader)
                acc_test_list.append(acc_test)
        print("time epoch : ", time.time() - start)


    # model save
    torch.save(model, config.path_save)

    # Loss 그래프
    plt.plot(step_list, loss_list, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    # Acc 그래프
    plt.plot(step_list, acc_test_list, 'b--')
    plt.legend(['Test acc', 'dev acc'])
    plt.show()

    print("test acc : ", acc_test_list)
    print("max test acc : ", max(acc_test_list))
    
