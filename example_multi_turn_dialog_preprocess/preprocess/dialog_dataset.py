import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch

from preprocess.data_read import get_data
from preprocess.sentence_piece_model import load_sentence_piece_model
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class dialog_dataset(Dataset):
    def __init__(self, cfg):
        self.dialog_dataset = get_data(cfg)
        self.spm = load_sentence_piece_model(cfg)
        self.spm.enable_truncation(max_length=cfg.max_seq)
        self.SEP = self.spm.token_to_id("<sep>")
        self.PAD = self.spm.token_to_id("<pad>")
        self.EOS = self.spm.token_to_id("<eos>")

    def __len__(self):
        return len(self.dialog_dataset)

    def index_to_sentence(self, index_dialog):
        index_dialog = index_dialog.numpy()  # torch -> numpy
        index_dialog = np.array(index_dialog, dtype=int)  # float -> int
        return self.spm.decode_batch(index_dialog)

    def __getitem__(self, idx):
        dialog = self.dialog_dataset[idx]
        index_dialog = []
        turn_id = []  # 발화간의 구분 id
        session_id = []  # 다수 발화와 마지막 발화간의 관계
        for turn_index, turn in enumerate(dialog):
            index_turn = self.spm.encode(turn).ids
            index_dialog.extend(index_turn)

            if turn_index % 2 == 0:
                turn_id.extend([0 for _ in range(len(index_turn) + 1)])
            else:
                turn_id.extend([1 for _ in range(len(index_turn) + 1)])

            if turn_index < len(dialog)-1:
                index_dialog.append(self.SEP)
                session_id.extend([0 for _ in range(len(index_turn)+1)])  # 1 을 더 붇인건 SEP count 때문
            else:
                index_dialog.append(self.EOS)
                session_id.extend([1 for _ in range(len(index_turn))])
                turn_id = turn_id[:-1]

        index_dialog = torch.Tensor(index_dialog).int()
        turn_id = torch.Tensor(turn_id).int()
        session_id = torch.Tensor(session_id).int()
        train = index_dialog[:-1]
        lm_label = index_dialog[1:]

        # print("index_dialog : ", len(index_dialog))
        # print("train : ", len(train))
        # print("lm_label : ", len(lm_label))
        # print("turn_id : ", len(turn_id))
        # print("session_id : ", len(session_id))

        return train, lm_label, turn_id, session_id

    @staticmethod
    def collate(batch):
        train = pad_sequence([i[0] for i in batch], batch_first=True, padding_value=0)
        lm_label = pad_sequence([i[1] for i in batch], batch_first=True, padding_value=0)
        turn_id = pad_sequence([i[2] for i in batch], batch_first=True, padding_value=3)
        session_id = pad_sequence([i[3] for i in batch], batch_first=True, padding_value=3)

        return (train, lm_label, turn_id, session_id)