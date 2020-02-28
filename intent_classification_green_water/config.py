import re

## hyper-parameter
epoch = 2 #10
batch = 1 # inference 시 1 로 변경할것
learning_rate = 0.0005

embedding_dim = 128 # 300
hidden_size = 128
linear_hidden_size = 128

## option
remove_outlier_seq = True
show_statistics = True # 전처리 할 때만 볼 수 있음

max_seq = 35
max_seq_cut = 35

## fixed
cpu_processor = 2
linear_dropout_keep_prob = 0.1
output_class = 7 #3
gpu = 'cpu' # 'cuda:0'
folder = "preprocessing/"
vocab_list = folder + "vocab_list.json"
word_to_index = folder + "word_to_index.json"
UNK = "[UNK]"
PAD = "[PAD]"
path_csv = "./train_data.csv"

regex = re.compile(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'\_…》]")


path_save = "./model.pth"