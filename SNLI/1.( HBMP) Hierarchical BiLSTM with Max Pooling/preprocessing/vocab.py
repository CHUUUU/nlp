import config as config
import pickle
import re
import collections
import os

class create_vocab():
    def __init__(self):
        self.UNK = config.UNK
        self.PAD = config.PAD
        self.labels = config.labels

    def __regular_expresion(self, sentence):
        clear = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
        return clear.lower()

    def __get_all_words_array(self, all_sentences):
        one_line_sentences = ' '.join(sentence for sentence in all_sentences)
        all_words_array = one_line_sentences.split(' ')
        return all_words_array

    def __remove_low_frequency_words(self, all_words_array):
        if config.use_remove_low_freq:
            unique = collections.Counter(all_words_array)  
            word_freq = unique.most_common() 
            length = len(word_freq)
            
            print("remove low frequency words")
            result = []
            low_freq_word = []
            for n, pair in enumerate(word_freq):
                word = pair[0]
                freq = pair[1]
                if freq < config.low_freq:
                    low_freq_word.append(word)
                    continue
                result.append(word)
            print("low_freq_word_count : ", len(low_freq_word))
            print("high_freq_word_count : ", len(result))

            string = ' '.join([str(i) for i in low_freq_word])
            f = open(config.low_freq_words_file , 'w') # 확인용
            f.write(string)
            f.close()
            print("낮은 빈도 단어 리스트가 저장 되었다.")
            return result
        return all_words_array

    def __indexing_vocab(self, all_words_array):
        self.vocab_list = [self.UNK, self.PAD] 
        self.vocab_list.extend(list(sorted(set(all_words_array))))
        self.word_to_index = { word:i for i,word in enumerate(self.vocab_list) }
        self.index_to_word = { i:word for i,word in enumerate(self.vocab_list) }

    def __save_words(self):
        with open(config.vocab_list, 'wb') as f:
            pickle.dump(self.vocab_list, f)
        print("vocab_list 가 저장되었다.")
        print("vocab_size : ", len(self.vocab_list))
        if not config.use_glove:
            with open(config.word_to_index, 'wb') as f:
                pickle.dump(self.word_to_index, f)
            print("word_to_index 가 저장되었다.")
            with open(config.index_to_word, 'wb') as f:
                pickle.dump(self.index_to_word, f)
            print("index_to_word 가 저장되었다.")

    def __load_words(self):
        with open(config.vocab_list, 'rb') as f:
            self.vocab_list = pickle.load(f) 
        print("vocab_list 를 불러왔다.")
        print("vocab_size : ", len(self.vocab_list))
        if not config.use_glove:
            with open(config.word_to_index, 'rb') as f:
                self.word_to_index = pickle.load(f) 
            print("word_to_index 를 불러왔다.")
            with open(config.index_to_word, 'rb') as f:
                self.index_to_word = pickle.load(f) 
            print("index_to_word 를 불러왔다.")

    def __main_flow(self):
        ################# 1. vocab 전처리 
        data = []
        all_sentences = []
        for path in self.path_list:
            with open(path, "r") as f:
                for n, line in enumerate(f):
                    if n == 0: # 첫줄은 무의미한 문장
                        continue
                    parts = line.strip().split("\t") # 한 줄당 여러 탭으로 파트가 나뉘어 져 있음
                    gold_label = parts[0] 
                    sentence1 = self.__regular_expresion(parts[5]) 
                    sentence2 = self.__regular_expresion(parts[6]) 
                    all_sentences.append(sentence1)
                    all_sentences.append(sentence2)

        all_words_array = self.__get_all_words_array(all_sentences)
        all_words_array = self.__remove_low_frequency_words(all_words_array)
        self.__indexing_vocab(all_words_array)

    def set_data(self, mode):
        if mode == config.all_mode:
            self.path_list = [config.path_train, config.path_test, config.path_dev]
        elif mode == config.train_mode:
            self.path_list = [config.path_train]
        elif mode == config.test_mode:
            self.path_list = [config.path_test]
        else:
            self.path_list = [config.path_dev]

        file_list = [config.vocab_list, config.word_to_index, config.index_to_word]
        for file in file_list:
            if os.path.isfile(file):
                exist = True
            else:
                exist = False
                break

        if exist: 
            self.__load_words()
        else:
            self.__main_flow()
            self.__save_words()
            

