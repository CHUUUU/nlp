import torch
import torch.nn as nn
import preprocessing.custom_dataset as custom_dataset
import model as model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config as config
import numpy as np
import matplotlib.pyplot as plt

def eval(loader):
    total = 0
    correct = 0
    for n, (label, sent1, sent2) in enumerate(loader):
        label = Variable(label.to(device))
        sent1 = Variable(torch.stack(sent1).to(device))
        sent2 = Variable(torch.stack(sent2).to(device))
        out =  model(sent1, sent2)
        _, pred = torch.max(out.data, 1)
        total += label.size(0) # batch size
        correct += (pred == label).sum() 
    acc = 100 * (correct.cpu().numpy()/total)
    return acc

if __name__ == "__main__":
    # 데이터 처리
    dataset = custom_dataset.Custom_dataset()
    train_data, test_data, dev_data = dataset.get_data()

    train_loader = DataLoader(train_data,
                            batch_size=config.batch ,
                            shuffle=True,
                            num_workers=config.cpu_processor,
                            drop_last=True)

    test_loader = DataLoader(test_data,
                            batch_size=config.batch,
                            shuffle=False,
                            num_workers=config.cpu_processor,
                            drop_last=True)

    dev_loader = DataLoader(dev_data,
                        batch_size=config.batch,
                        shuffle=False,
                        num_workers=config.cpu_processor,
                        drop_last=True)

    # 모델 설정
    device = torch.device(config.gpu if torch.cuda.is_available() else 'cpu')
    model = model.Classifier(dataset.vocab_list)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    print("--model set--")

    # 훈련
    step_list = []
    loss_list = []
    acc_test_list = []
    acc_dev_list = []
    step = 0
    for i in range(config.epoch):
        print("epoch = ", i)
        for n, (label, sent1, sent2) in enumerate(train_loader):
            optimizer.zero_grad() # 초기화
            label = Variable(label.to(device))
            sent1 = Variable(torch.stack(sent1).to(device))
            sent2 = Variable(torch.stack(sent2).to(device))
            logit = model(sent1, sent2)
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            step += 1
            if n % 500 == 0:
                print("epoch : ", i, " step : ", n , " loss : ", loss.item())
                step_list.append(step)
                loss_list.append(loss)
                acc_test = eval(test_loader)
                acc_dev = eval(dev_loader)
                acc_test_list.append(acc_test)
                acc_dev_list.append(acc_dev)
    
    # Loss 그래프
    plt.plot(step_list, loss_list, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    # Acc 그래프
    plt.plot(step_list, acc_test_list, 'b--')
    plt.plot(step_list, acc_dev_list, 'g--')
    plt.legend(['Test acc', 'dev acc'])
    plt.show()

    print("test acc : ", acc_dev_list)
    print("test acc : ", acc_test_list)

    
