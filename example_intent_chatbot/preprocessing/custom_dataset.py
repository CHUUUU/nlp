import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils import data
import config as c
import pandas as pd
from preprocessing.sent_piece_model import sentence_piece_model
import random

class dummy_dataset(data.Dataset):
    def __init__(self, custom_dataset):        
        self.dataset = custom_dataset
        self.len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len

class Custom_dataset():
    def __init__(self):
        self.stat_length_list = []
        self.spm = sentence_piece_model()
        self.spm.train_or_load()

    def __word_count_statistics(self):
        for i in range(max(self.stat_length_list)):
            print(i+1, " : ",  self.stat_length_list.count(i+1))
      
    def __label_count_statistics(self, all_label):
        for label in c.labels:
            print(label, " : ",  all_label.count(label))

    def __main_flow(self):
        custom_dataset = []
        with open(c.data_csv_path, newline='') as csvfile:
            df = pd.read_csv(csvfile)

            user_question = df['Q'].values.tolist()
            # bot_answer = df['A'].values.tolist()
            label = df['label'].values.tolist()

            for n in range(len(user_question)):
                q = user_question[n]
                l = label[n]

                new_q = self.spm.convert_word_to_index(q)
                self.stat_length_list.append(len(new_q))
                new_q = self.spm.padding(new_q)
                new_q = new_q[:c.max_seq]

                custom_dataset.append((new_q, l))

        if c.show_statistics:
            self.__word_count_statistics()
            self.__label_count_statistics(label)

        random.seed(7)
        random.shuffle(custom_dataset)
        return custom_dataset

    def get_data(self):
        data = self.__main_flow()
        data = dummy_dataset(data)
        print("dataset ready")
        return data