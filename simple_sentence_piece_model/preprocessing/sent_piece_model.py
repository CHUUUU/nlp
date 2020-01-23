import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import config as c
import sentencepiece as spm
from sentencepiece import SentencePieceTrainer
from preprocessing.csv_to_txt import convert_csv_to_txt


class sentence_piece_model():
    def __init__(self): # spm = sentence_piece_model
        if not os.path.isfile(c.data_text_path):
            convert_csv_to_txt()

    def __spm_create(self):
        if os.path.isfile("data/love.model"):
            return 0

        params = '--input=' + c.data_text_path + \
                 ' --model_prefix=data/love ' \
                 ' --vocab_size=2500' \
                 ' --model_type=' + c.model_type[0] + \
                 ' --max_sentence_length=999999' \
                 ' --pad_id=0 --pad_piece=[PAD]' \
                 ' --unk_id=1 --unk_piece=[UNK]' \
                 ' --character_coverage=1.0'
        # 0.9995 for english, 1.0 for Korean
        SentencePieceTrainer.Train(params)


    def __spm_load(self):
        s = spm.SentencePieceProcessor()
        s.Load('data/love.model')
        return s

    def train(self):
        self.__spm_create()
        self.sp_model = self.__spm_load()

    def convert_word_to_index(self, word_list):
        # word_list = s.encode_as_pieces(line)
        index_list = self.sp_model.encode_as_ids(word_list)
        # print(word_list)
        # print(index_list)
        return index_list