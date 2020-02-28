import re
import torch

# dataset
# https://github.com/songys/Chatbot_data

## hyper-parameter
epoch = 10
batch = 1
learning_rate = 0.0005

embedding_dim = 300
factorization_emb_dim = 128
hidden_size = 128

linear_hidden_size = 128

## option
remove_outlier_seq = True
show_statistics = True  # 전처리 할 때만 볼 수 있음
max_seq = 29

## fixed
vocab_size = 2507
cpu_processor = 2
linear_dropout_keep_prob = 0.1
output_class = 3
folder = "preprocessing/"
# vocab_list = folder + "vocab_list.json"
word_to_index = folder + "word_to_index.json"
data_csv_path = "data/ChatbotData.csv"
data_text_path = "data/spm_train_data.txt"
model_path = "model.pth"
UNK = "[UNK]"
PAD = "[PAD]"

model_type = ['unigram', 'bpe', 'char', 'word']
labels = [0, 1, 2]
# 0 = 일상다반사
# 1 = 이별 (부정)
# 2 = 사랑 (긍정)


regex = re.compile(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'\_…》]")