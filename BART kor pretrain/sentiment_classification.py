import torch
import torch.nn as nn
import transformer
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

max_seq = 256

class binary_classification(nn.Module):
    def __init__(self, bart_model=None, freeze_bart=True, vocab_size=None):
        super(binary_classification, self).__init__()

        self.bart = bart_model
        self.vocab_size = vocab_size

        if freeze_bart:
            for p in self.bart.parameters():
                p.requires_grad = False
        
        self.cls_layer = nn.Linear(self.vocab_size, 1)

    def forward(self, ko_enc, ko_dec, last_token_position, batch_size):
        with torch.no_grad():
            bart_logit = self.bart(ko_enc, ko_dec)   # bart_logit.shape : [8160, 32000] -> [batch(32) * (256-1)255, vocab_size]\
            bart_logit = bart_logit.view(batch_size, max_seq-1, self.vocab_size)  # bart_logit.shape : torch.Size([32, 255, 32000])
        
        final_logit = torch.empty(size=(batch_size, self.vocab_size)).cuda()  # torch.Size([32, 32000])
        for i in range(batch_size):
            final_logit[i] = bart_logit[i, last_token_position[i]]

        cls_logit = self.cls_layer(final_logit)  # cls_label.shape :  torch.Size([32])
        return cls_logit


def get_dataset(spm, paths):
    index_lines = []
    labels = []
    last_token_position = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            all_lines = f.readlines()
            for index, line in enumerate(tqdm(all_lines)):
                if index == 0:
                    continue
                # id	document	label
                # 7797314	원작의 긴장감을 제대로 살려내지못했다.	0
                split = line.split('\t')
                doc_id = split[0]
                line = spm.encode(split[1]).ids
                label = int(split[2][0].strip('\n'))

                if len(line) > max_seq:  # max seq = 256
                    line = line[:max_seq]

                index_lines.append(torch.tensor(line))
                labels.append(label)
                last_token_position.append(len(line)-1)

    index_lines_pad = pad_sequence(index_lines, batch_first=True, padding_value=0)
    return index_lines_pad, labels, last_token_position

class nsmc_dataset(Dataset):
    def __init__(self, ko_spm, nsmc_path):
        self.ko_train_data, self.cls_label, self.last_token_position = get_dataset(ko_spm, nsmc_path)

    def __getitem__(self, idx):
        ko_enc_input = self.ko_train_data[idx]
        ko_dec_input = self.ko_train_data[idx, :-1]
        ko_dec_target = self.cls_label[idx]
        last_token_position = self.last_token_position[idx]

        return ko_enc_input, ko_dec_input, ko_dec_target, last_token_position

    def __len__(self):
        return len(self.ko_train_data)

# import create_spm.spm as spm
# ko_spm, en_spm = spm.get_spm()
# ko_vocab_size = ko_spm.get_vocab_size()
# a = nsmc_dataset(ko_spm)