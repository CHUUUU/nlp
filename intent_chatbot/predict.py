import torch
from torch.autograd import Variable
import config as c
from preprocessing.sent_piece_model import sentence_piece_model
import os
import model as model
from torch.utils.data import DataLoader
import random

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = model.classifier()
    model.load_state_dict(torch.load(c.model_path))
    model.eval()
    model.cuda()

    spm = sentence_piece_model()
    spm.train_or_load()

    c.batch_size = 1
    
    print("안녕하세요 ㅎㅎ")

    while True:
        q = input("\n")
        q = spm.convert_word_to_index(q)
        q = spm.padding(q)
        q = q[:c.max_seq]

        predict_loader = DataLoader([q], shuffle=False)
        for q in predict_loader:
            q = Variable(torch.stack(q).cuda())
            out = model(q)
            _, pred = torch.max(out.data, 1)
            pred = pred.tolist()[0]
            # print(pred)

            if pred == 0:
                a = ["아마도요!", "그렇겠죠?", "나 말고 경훈이 한테 말해요", "난 그냥 막걸리나 마셨으면 좋겠다"]
            elif pred == 1:
                a = ["아 이런 ㅜㅜ", "힘내요ㅜ"]
            elif pred == 2:
                a = ["할렐루야!", "아 역시 될 줄 알았어 ㅎㅎ", "축하염ㅎㅎ", "이런 날엔 소고기지!!"]

            print(random.choice(a))


