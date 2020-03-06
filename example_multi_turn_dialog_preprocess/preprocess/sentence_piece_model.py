import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import Config
import warnings
from preprocess.data_read import get_data_file_path_list

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

warnings.filterwarnings("error")
cfg = Config.load()

def save_sentense_piece_model():
    paths = get_data_file_path_list()

    special_token = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>", "<mask>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=32000, min_frequency=2, special_tokens=special_token)
    tokenizer.save(cfg.path_sentence_piece, "ko")

def load_sentence_piece_model():
    tokenizer = ByteLevelBPETokenizer(cfg.path_sentence_piece_vocab, cfg.path_sentence_piece_model)
    return tokenizer

if __name__ == "__main__":
    save_sentense_piece_model()
    tokenizer = load_sentence_piece_model()

    tokenizer.enable_truncation(max_length=512)
    encoding = tokenizer.encode("콜")
    print(encoding.tokens)
    print(encoding.special_tokens_mask)
    print(encoding.ids)
    print(encoding.normalized_str)

    # ['ì', '½ľ']
    # [0, 0]
    # [174, 517]
    # 콜

    decode = tokenizer.decode_batch([[174, 517]])
    print(decode)
    # 콜
