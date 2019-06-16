## hyper-parameter
epoch = 5
batch = 256
learning_rate = 0.0005

embedding_dim = 600
hidden_size = 128
linear_hidden_size = 128

## option
use_stop_word = False
use_UNK = False
use_remove_low_freq = False

low_freq = 2
max_seq = 21
min_seq_cut = 3
max_seq_cut = 29

## fixed
cpu_processor = 2
linear_dropout_keep_prob = 0.1
output_class = 3
gpu = 'cuda:0'
folder = "preprocessing/"
preprocessed_train_file = folder + "preprocessed_train_data.pickle"
preprocessed_test_file = folder + "preprocessed_test_data.pickle"
preprocessed_dev_file = folder + "preprocessed_dev_data.pickle"
low_freq_words_file = folder + "low_freq_words_file.txt"
vocab_list = folder + "vocab_list.pickle"
vocab_size = folder + "vocab_size.pickle"
word_to_index = folder + "word_to_index.pickle"
index_to_word = folder + "index_to_word.pickle"
path = "../../data/snli_1.0/"
path_train = path + "snli_1.0_train.txt"
path_test = path + "snli_1.0_test.txt"
path_dev = path + "snli_1.0_dev.txt"
labels = ["neutral", "contradiction", "entailment"] 
UNK = "[UNK]"
PAD = "[PAD]"

train_mode = 0
test_mode = 1 
dev_mode = 2
all_mode = 3