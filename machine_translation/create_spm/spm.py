from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def save_sentense_piece_model():
    ko_paths = ['./data/korean-english-park.dev.ko', './data/korean-english-park.train.ko']
    en_paths = ['./data/korean-english-park.dev.en', './data/korean-english-park.train.en']

    special_token = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=ko_paths, vocab_size=32000, min_frequency=2, special_tokens=special_token)
    tokenizer.save("./create_spm", "ko")

    tokenizer.train(files=en_paths, vocab_size=32000, min_frequency=2, special_tokens=special_token)
    tokenizer.save("./create_spm", "en")

def load_sentence_piece_model(path_vocab, path_model):
    tokenizer = ByteLevelBPETokenizer(path_vocab, path_model)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>"))
    )

    tokenizer.enable_truncation(max_length=512)

    # encoding = tokenizer.encode("배고파요")
    # print(encoding.tokens)
    # print(encoding.special_tokens_mask)
    # print(encoding.ids)
    # print(encoding.normalized_str)
    #
    # decoding = tokenizer.decode([2, 1177, 276, 692, 571, 1])
    # print(decoding)

    return tokenizer

def get_spm():
    path_model = "./create_spm"
    ko_path_vocab = path_model + "/ko-vocab.json"
    ko_path_model = path_model + "/ko-merges.txt"
    en_path_vocab = path_model + "/ko-vocab.json"
    en_path_model = path_model + "/ko-merges.txt"
    ko_tokenizer = load_sentence_piece_model(ko_path_vocab, ko_path_model)
    en_tokenizer = load_sentence_piece_model(en_path_vocab, en_path_model)

    # encoding = ko_tokenizer.encode("배고파요")
    # print(encoding.tokens)
    # print(encoding.special_tokens_mask)
    # print(encoding.ids)
    # print(encoding.normalized_str)
    #
    # decoding = ko_tokenizer.decode([2, 1177, 276, 692, 571, 1])
    # print(decoding)  # <eos>배고파요<bos>

    return ko_tokenizer, en_tokenizer

if __name__ == "__main__":
    # save_sentense_piece_model()
    get_spm()


# ['<eos>', 'ë°°', 'ê³ł', 'íĮĮ', 'ìļĶ', '<bos>']
# [1, 0, 0, 0, 0, 1]
# [2, 1177, 276, 692, 571, 1]
# 배고파요
