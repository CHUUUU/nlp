import config as config
import re
import collections
import os
import json
import csv
import pickle

class create_vocab():
    def __init__(self, path_csv):
        self.regex = config.regex
        self.path_csv = path_csv

    def __save_words(self):
        with open(config.vocab_list, 'w') as f:
            json.dump(self.__vocab_list, f)

    def __load_words(self):
        with open(config.vocab_list, 'r') as f:    
            self.__vocab_list = json.load(f)

    def __main_flow(self):
        all_sentences = []

        with open(self.path_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            index = 0
            for line in reader:
                if index == 0:
                    index += 1
                    continue
                if line[0] == '':
                    continue
                sentence = self.regex.sub('', line[1])  
                all_sentences.append(sentence.lower())

        one_line = ' '.join(sentence for sentence in all_sentences)
        all_words = one_line.split(' ')
        
        self.__vocab_list = [config.UNK, config.PAD] 
        self.__vocab_list.extend(list(sorted(set(all_words))))
        self.__word_to_index = { word:i for i,word in enumerate(self.__vocab_list) }

    def get_data(self):
        self.__main_flow()
        print("vocab_size : ", len(self.__vocab_list))
        self.__save_words()
        return self.__word_to_index

