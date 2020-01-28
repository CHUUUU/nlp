import torch
import preprocessing.custom_dataset as c
import preprocessing.vocab as v
from torch.autograd import Variable
import config

class predict():
    def __init__(self):
        path = "./train_data.csv"
        vocab = v.create_vocab(path_csv=path)
        word_to_index = vocab.get_data()
        self.dataset = c.Custom_dataset(word_to_index, path)
        self.model = torch.load("./model.pth")
        

    def predict(self, string_sentence):
        index_array, _ = self.dataset.preprocessing(string_sentence)
        print(index_array)
        a = torch.tensor(index_array)
        a = a.view(config.max_seq, -1)
        b = a.to('cpu')
        sent = Variable(b)
        print(sent)
        
        self.model.eval() # 확인용
        output = self.model(sent)#sent)
        _, pred = torch.max(output.data, 1)
        return pred # class index

if __name__ == "__main__" :
    p = predict()
    pred = p.predict("녹물이 너무 나와요")
    print(pred)