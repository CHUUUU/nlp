from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

path_model = "./sentence_piece_model/"
path_vocab = path_model + "/ko-vocab.json"
path_model = path_model + "/ko-merges.txt"

def save_sentense_piece_model():
    paths = [str(x) for x in Path("./data/").glob("**/*.txt")]
    print(paths)

    special_token = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=32000, min_frequency=2, special_tokens=special_token)
    tokenizer.save(".", "ko")

def load_sentence_piece_model():
    tokenizer = ByteLevelBPETokenizer(path_vocab, path_model)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>"))
    )

    tokenizer.enable_truncation(max_length=512)
    encoding = tokenizer.encode("배고파요")
    print(encoding.tokens)
    print(encoding.special_tokens_mask)
    print(encoding.ids)
    print(encoding.normalized_str)

if __name__ == "__main__":
    # save_sentense_piece_model()
    load_sentence_piece_model()





