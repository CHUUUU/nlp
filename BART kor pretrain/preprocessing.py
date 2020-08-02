import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import copy


####################
# BART
####################

# 이 아키텍처는
# - BERT에 사용된 아키텍처와 밀접한 관련이 있으며 다음과 같은 차이점이 있습니다.
# - (1) 디코더의 각 계층은
#     - (transformer시퀀스-시퀀스 모델에서와 같이) 인코더의 final hidden layer 에 대해 cross-attention 를 추가로 수행하고;
# - (2) BERT는
#     - 단어 예측 전에 추가 피드 포워드 네트워크를 사용하지만
#     - BART에서는 그렇지 않습니다.

# BART는
# - 문서를 손상시킨 다음
# - 디코더의 출력과 원본 문서 사이의 교차 엔트로피인 재구성 손실을 최적화함으로써 학습됩니다.


# Poisson 분포 (λ = 3)에서 추출한 범위 길이를 사용하여
# 여러 텍스트 span 이 샘플링됩니다.
# 각 Span 은 단일 [MASK] 토큰으로 대체됩니다.
# 길이가 0 span인 는 [MASK] 토큰 삽입에 해당합니다. (ABC, DE) → length 0 → ABC, D [mask] E
#
# 텍스트 채움(Text infilling)은
# - SpanBERT (Joshi et al., 2019)에서 영감을 얻었지만
# - SpanBERT 샘플은 길이가 다른 분포(클램프된 기하학적(clamped geometric))와 길이가 같으며 각 범위를 정확히 동일한 길이의 [MASK] 토큰 시퀀스로 대체합니다.
#
# 텍스트 채움은
# - 모델이 범위에서 누락된 토큰 수를 예측하도록 지시합니다.

###################
# span bert
###################
# 일련의 토큰 X = (x1,..., xn)이 주어지면
# 마스크 예산 (예 : X의 15 %)이 소비 될 때까지
# 텍스트 범위를 반복적으로 샘플링하여
# 토큰 Y ⊆ X의 하위 집합을 선택합니다.

# 반복 할 때마다 **먼저 기하 분포(geometric distribution) L ~ Geo (p)에서 스팬 길이를 샘플링**합니다.
# 이 길이는 더 짧은 스팬쪽으로 기울어졌습니다.
# 그런 다음 스팬의 시작점을 무작위로 (균일하게) 선택합니다
# BERT와 마찬가지로 토큰의 15 %를 마스킹합니다.
# 마스크 된 토큰의 80 %를 [MASK]로,
# 10 %는 임의 토큰으로,
# 10 %는 원래 토큰으로 대체합니다.
# 그러나 이 교체는 스팬 수준에서 수행되며
# 각 토큰마다 개별적으로 수행되지는 않습니다.
# 즉, 스팬의 모든 토큰은 [MASK] 또는 샘플링 된 토큰으로 대체됩니다.


def create_train_index_line(index_line, mask_token_index):
    train_index_line = copy.deepcopy(index_line)
    count_mask_token = 0
    for i, token in enumerate(index_line):
        prob = random.random()
        if prob < 0.15:
            span_length = np.random.poisson(lam=3, size=1)[0]

            # span 이 0 이면 i + 1 position 에 mask token 삽입하기
            if span_length == 0:
                train_index_line.insert(i+1, mask_token_index)

            # span 이 0 이상이면, single mask token 으로 대체
            if i >= len(train_index_line):
                break

            train_index_line[i] = mask_token_index

            for j in range(1, span_length):
                if int(len(index_line) * 0.15) <= count_mask_token:
                    break
                if i+1 >= len(train_index_line):
                    break

                count_mask_token += 1
                # print("############################")
                # print("span_length : ", span_length)
                # print("count_mask_token : ", count_mask_token)
                # print("i+1 : ", i+1)
                # print("train_index_line : ", train_index_line)
                # print("len(train_index_line) : ", len(train_index_line))
                # print("train_index_line(i+1) : ", train_index_line[i + 1])
                del train_index_line[i+1] # span-1 길이만큼 remove

    return train_index_line


def get_index(spm, paths, mask_token_index):
    index_lines = []
    train_index_line = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                id_line = spm.encode(line).ids
                index_lines.append(torch.tensor(id_line))
                train_index_line.append(torch.tensor(create_train_index_line(id_line, mask_token_index)))

    # print("index_lines_pad.shape : ", index_lines.shape)
    # print("train_index_line.shape : ", train_index_line.shape)

    index_lines_pad = pad_sequence(index_lines, batch_first=True, padding_value=0)
    train_index_line_pad = pad_sequence(train_index_line, batch_first=True, padding_value=0)

    # print("index_lines_pad.shape : ", index_lines.shape)
    # print("index_lines_pad.shape : ", train_index_line_pad.shape)

    return index_lines_pad, train_index_line_pad

class custom_dataset(Dataset):
    # def __init__(self, ko_spm, en_spm, ko_paths, en_paths):
    def __init__(self, ko_spm, ko_paths, mask_token_index):
        self.ko_dataset, self.ko_dataset_train = get_index(ko_spm, ko_paths, mask_token_index)
        # self.en_dataset = get_index(en_spm, en_paths)

    def __getitem__(self, idx):
        ko_enc_input = self.ko_dataset_train[idx]
        ko_dec_input = self.ko_dataset[idx, :-1]
        ko_dec_target = self.ko_dataset[idx, 1:]
        # en_dec_input = self.en_dataset[idx, :-1]
        # en_dec_target = self.en_dataset[idx, 1:]

        return ko_enc_input, ko_dec_input, ko_dec_target

    def __len__(self):
        return len(self.ko_dataset)