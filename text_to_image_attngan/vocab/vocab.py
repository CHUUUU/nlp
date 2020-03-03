import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config.config as c
import util as u

class vocab():
    def __init__(self):
        self.__vocab_list = [c.EOS_TOKEN, c.PAD_TOKEN]

    def create(self, text_lines):
        all_word = []
        for sent in text_lines:
            for word in sent:
                all_word.append(word)

        self.__vocab_list.extend(list(sorted(set(all_word))))
        word_to_index = {word: i for i, word in enumerate(self.__vocab_list)}
        u.json_save(c.vocab_path, word_to_index)


