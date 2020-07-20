import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def get_index(spm, paths):
    index_lines = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                index_lines.append(torch.tensor(spm.encode(line).ids))
    index_lines_pad = pad_sequence(index_lines, batch_first=True, padding_value=0)
    print("index_lines_pad.shape : ", index_lines_pad.shape)
    return index_lines_pad

class custom_dataset(Dataset):
    def __init__(self, ko_spm, en_spm, ko_paths, en_paths):
        self.ko_dataset = get_index(ko_spm, ko_paths)
        self.en_dataset = get_index(en_spm, en_paths)

    def __getitem__(self, idx):
        ko_enc_input = self.ko_dataset[idx]
        en_dec_input = self.en_dataset[idx, :-1]
        en_dec_target = self.en_dataset[idx, 1:]

        return ko_enc_input, en_dec_input, en_dec_target

    def __len__(self):
        return len(self.ko_dataset)