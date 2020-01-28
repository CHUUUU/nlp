import torch
import numpy as np
import config

def get_sinusoid_encoding_table(n_position):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / config.embedding_dim)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(config.embedding_dim)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

