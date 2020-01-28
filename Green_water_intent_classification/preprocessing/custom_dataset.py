import re
import numpy as np
import config as config
import os
import torch.utils.data as data
import csv

class dummy_dataset(data.Dataset):
    def __init__(self, custom_dataset):        
        self.dataset = custom_dataset
        self.len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len

class Custom_dataset():
    def __init__(self, word_to_index, path_csv):
        self.word_to_index = word_to_index
        self.path_csv=path_csv
        self.vocab_list = list(word_to_index.keys())
        self.regex = config.regex

    def __PAD(self, index_array):
        while len(index_array) < config.max_seq_cut:
            index_array.append(1) # PAD_INDEX
        return index_array

    def __find_missing_word(self, word):
        if word not in self.vocab_list: 
            return config.UNK
        else:
            return word

    def __convert_word_to_index(self, word):
        word = self.__find_missing_word(word)
        word = self.word_to_index[word]
        return word

    def __word_count_statistics(self, orgin_max_length):
        for i in range(max(orgin_max_length)): 
            print(i+1, " : ",  orgin_max_length.count(i+1))
      
    def __label_count_statistics(self, all_label):
        for label in [0, 1, 2, 3, 4, 5, 6]: 
            print(label, " : ",  all_label.count(label))

    def preprocessing(self, sentence):
        sentence = self.regex.sub('', sentence.lower()) 
        word_array = sentence.split(' ')
        index_array = [self.__convert_word_to_index(word) for word in word_array]
        length = len(index_array)
        index_array = self.__PAD(index_array)
        index_array = index_array[:config.max_seq]
        return index_array, length

    def __main_flow(self):
        stat_label =[]
        index_data = []
        stat_length = []
        with open(self.path_csv, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            index = 0
            for line in reader:
                if index == 0:
                    index += 1
                    continue
                if line[0] == '':
                    continue
                question = line[1].lower()
                label = int(line[0]) - 1
                index_sent, length = self.preprocessing(question)

                index_data.append((label, index_sent))
                stat_length.append(length)
                stat_label.append(label)
     
        if config.show_statistics:
            self.__word_count_statistics(stat_length)
            self.__label_count_statistics(stat_label)
        return index_data


    def get_data(self):
        data = self.__main_flow()
        index_train_data = dummy_dataset(data)

        return index_train_data