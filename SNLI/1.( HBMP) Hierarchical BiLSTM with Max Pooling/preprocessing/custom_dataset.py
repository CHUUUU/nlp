import re
import torch
import numpy as np
import config as config
import nltk
import os
import pickle
import preprocessing.vocab as vocab
from nltk.corpus import stopwords  
from torch.utils import data

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
        self.path_train = config.path_train
        self.path_test = config.path_test
        self.path_dev = config.path_dev

        custom_vocab = vocab.create_vocab()
        custom_vocab.set_data(mode=config.vocab_mode)
        self.vocab_list = custom_vocab.vocab_list
        self.word_to_index = custom_vocab.word_to_index
        self.index_to_word = custom_vocab.index_to_word

        self.stop_words = stopwords.words('english')

        self.labels = config.labels
        self.label_to_index = {
            self.labels[0] : 0,  
            self.labels[1] : 1, 
            self.labels[2] : 2
            }

    def __stop_words(self, word_array):
        if config.use_stop_word:
            print("apply stop word")
            result = [] 
            for w in word_array: 
                if w not in self.stop_words: 
                    result.append(w) 
            return result
        return word_array

    def __regular_expresion(self, sentence):
        clear = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
        return clear.lower()

    def __find_missing_word(self, word):
        if [word] not in self.no_duplication_words: # 찾는 형태가 배열이여야지 찾음
            return config.UNK
        else:
            return word

    def __UNK(self, str_train_data):
        if config.use_UNK:
            print("--apply UNK--")
            result = []
            length = len(str_train_data)
            for n, (gold_label, word_array1, word_array2) in enumerate(str_train_data):
                if n % 3000 == 0:
                    print((n*100)/length,"%")
                new_word_array1 = [self.__find_missing_word(word) for word in word_array1]
                new_word_array2 = [self.__find_missing_word(word) for word in word_array2]   
                result.append((gold_label, new_word_array1, new_word_array2))
            print("100%")
            return result
        return str_train_data

    def __PAD(self, index_array):
        while len(index_array) < config.max_seq:
            index_array.append(1) # PAD_INDEX
        return index_array

    def __Max_length(self, index_array):
        if len(index_array) > config.max_seq:
            index_array = index_array[:config.max_seq]
        return index_array

    def __word_count_statistics(self, orgin_max_length):
        for i in range(max(orgin_max_length)): # 78
            print(i+1, " : ",  orgin_max_length.count(i+1))
      
    def __label_count_statistics(self, all_label):
        for label in self.labels: 
            print(label, " : ",  all_label.count(label))

    def __main_flow(self, path, mode):
        if mode ==  config.train_mode:
            print("--train data 전처리를 시작합니다--")
        elif mode ==  config.test_mode:
            print("--test data 전처리를 시작합니다--")
        else:
            print("--dev data 전처리를 시작합니다--")

        ################# 2. 문자열 전처리 (split, regex, stop word, UNK)
        data = []
        stat_label =[]
        with open(path, "r") as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                parts = line.strip().split("\t") # 한 줄당 여러 탭으로 파트가 나뉘어 져 있음
                gold_label = parts[0] 
                sentence1 = self.__regular_expresion(parts[5]) 
                sentence2 = self.__regular_expresion(parts[6]) 
                word_array1 = sentence1.split(' ')
                word_array2 = sentence2.split(' ')
                word_array1 = self.__stop_words(word_array1)
                word_array2 = self.__stop_words(word_array2)
                stat_label.append(gold_label)

                data.append((gold_label, word_array1, word_array2))

        print("--Done sentence to word array--")
        data = self.__UNK(data)

        ################# 3. 인덱스 전처리 (indexing, PAD, Max_Length)
        print("--apply indexing, PAD, Max_Length--")
        index_data = []
        stat_length = []
        for n, data_tuple in enumerate(data):
            (gold_label, word_array1, word_array2) = data_tuple
            if gold_label not in self.labels: # 레이블 없는 것 예외처리
                continue
            if not config.use_outlier_sentence:    
                if len(word_array1) < config.min_seq_cut:
                    continue
                if len(word_array2) < config.min_seq_cut:
                    continue
                if len(word_array1) > config.max_seq_cut:
                    continue
                if len(word_array2) > config.max_seq_cut:
                    continue
            
            index_label = self.label_to_index[gold_label]
            index_sent1 = [self.word_to_index[word] for word in word_array1]
            index_sent2 = [self.word_to_index[word] for word in word_array2]
            stat_length.append(len(index_sent1))
            stat_length.append(len(index_sent2))

            # max_len 모자란 길이 PAD 입력
            index_sent1 = self.__PAD(index_sent1)
            index_sent2 = self.__PAD(index_sent2)

            # max_len 넘는 배열 제거
            index_sent1 = self.__Max_length(index_sent1)
            index_sent2 = self.__Max_length(index_sent2)

            index_data.append((index_label, index_sent1, index_sent2))
     
        print("--전처리가 끝났습니다--")
        if config.show_statistics:
            self.__word_count_statistics(stat_length)
            self.__label_count_statistics(stat_label)
        return index_data

    def __load_or_preprocessing(self, mode):
        if mode ==  config.train_mode:
            path = self.path_train
            file_name = config.preprocessed_train_file
        elif mode == config.test_mode:
            path = self.path_test
            file_name = config.preprocessed_test_file
        else:
            path = self.path_dev
            file_name = config.preprocessed_dev_file

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                data = pickle.load(f) # 전처리된 데이터
        else:
            data = self.__main_flow(path, mode)
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
        return data

    def get_data(self):
        index_train_data = self.__load_or_preprocessing(config.train_mode)
        index_test_data = self.__load_or_preprocessing(config.test_mode)
        index_dev_data = self.__load_or_preprocessing(config.dev_mode)

        index_train_data = dummy_dataset(index_train_data)
        index_test_data = dummy_dataset(index_test_data)
        index_dev_data = dummy_dataset(index_dev_data)

        return (index_train_data, index_test_data, index_dev_data)