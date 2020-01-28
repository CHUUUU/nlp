import torch
import torch.nn as nn
import preprocessing.custom_dataset as custom_dataset
import model as model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config as c
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def eval(loader):
    total = 0
    correct = 0
    for (q, label) in loader:
        label = Variable(label.cuda())
        q = Variable(torch.stack(q).cuda())
        out = model(q)
        _, pred = torch.max(out.data, 1)
        total += label.size(0)  # batch size
        correct += (pred == label).sum()
    acc = 100 * (correct.cpu().numpy()/total)
    return acc

if __name__ == "__main__":
    # 데이터 처리
    assert torch.cuda.is_available(), "cuda is not available"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dataset = custom_dataset.Custom_dataset()
    train_data = dataset.get_data()

    val_data = train_data[:100]
    train_data = train_data[100:]

    train_loader = DataLoader(train_data,
                            batch_size=c.batch,
                            shuffle=True,
                            num_workers=1,#.cpu_processor,
                            drop_last=True)

    # test_loader = DataLoader(test_data,
    #                         batch_size=c.batch,
    #                         shuffle=False,
    #                         num_workers=c.cpu_processor,
    #                         drop_last=True)

    dev_loader = DataLoader(val_data,
                        batch_size=c.batch,
                        shuffle=False,
                        num_workers=c.cpu_processor,
                        drop_last=True)

    # 모델 설정
    model = model.classifier()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 훈련
    step_list = []
    loss_list = []
    acc_test_list = []
    acc_dev_list = []
    step = 0
    for i in range(c.epoch):
        start = time.time()
        for n, (q, label) in enumerate(train_loader):
            optimizer.zero_grad() # 초기화
            label = Variable(label.cuda())
            q = Variable(torch.stack(q).cuda())
            logit = model(q)

            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            step += 1
            if n % 500 == 0:
                step_list.append(step)
                loss_list.append(loss)
                # acc_test = eval(test_loader)
                # acc_test_list.append(acc_test)
                acc_dev = eval(dev_loader)
                acc_dev_list.append(acc_dev)
                print("epoch: ", i, "  step: ", step, "  loss: ", loss.item(), "  time : ", time.time() - start, "  acc_dev: ", acc_dev)

    torch.save(model.state_dict(), c.model_path)
    print("model saved")


    # # Loss 그래프
    # plt.plot(step_list, loss_list, 'r--')
    # plt.legend(['Training Loss'])
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # # Acc 그래프
    # # plt.plot(step_list, acc_test_list, 'b--')
    # plt.plot(step_list, acc_dev_list, 'g--')
    # plt.legend(['Test acc', 'dev acc'])
    # plt.show()

    # print("test acc : ", acc_test_list)
    # print("max test acc : ", max(acc_test_list))

    print("dev acc : ", acc_dev_list)
    print("max dev acc : ", max(acc_dev_list))



