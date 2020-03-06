import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preprocess.data_read import get_file_path_list
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


def save_sentense_piece_model(cfg):
    paths = get_file_path_list(cfg)
    special_token = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>", "<mask>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=32000, min_frequency=2, special_tokens=special_token)
    tokenizer.save(cfg.path_sentence_piece, "ko")

def load_sentence_piece_model(cfg):
    if cfg.save_spm_model:
        save_sentense_piece_model(cfg)
    tokenizer = ByteLevelBPETokenizer(cfg.path_sentence_piece_vocab, cfg.path_sentence_piece_model)
    test(tokenizer)
    return tokenizer

def test(tokenizer):
    # test
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