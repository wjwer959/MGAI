import torch
import numpy as np
from datetime import datetime
import random

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def normalization(data):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    norm = np.zeros((2, data.shape[1]))
    data_new = np.zeros(data.shape)
    for i in range(data.shape[1]):
        ma, mi = max(data[:, i]), min(data[:, i])
        norm[0, i], norm[1, i] = ma, mi
        for j in range(data.shape[0]):
            data_new[j, i] = (data[j, i] - mi) / (ma - mi)
    return data_new, norm


def inormalization(data, norm):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] != norm.shape[1]:
        raise ZeroDivisionError('')
    data_new = np.zeros(data.shape)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            data_new[j, i] = data[j, i] * (norm[0, i] - norm[1, i]) + norm[1, i]
    return data_new

def par_inspect(coding_name, shifty_par):
    mode_dic = {'local_coding_mode': [0, 1, '0', '1'], 'global_coding_mode': [0, 1, 2, 3, '0', '1', '2', '3']}

    if type(shifty_par[coding_name]) == list:
        for j in range(len(shifty_par[coding_name])):
            if shifty_par[coding_name][j] not in mode_dic[coding_name]:
                raise ZeroDivisionError('parameter \"' + coding_name + '\" must in ' + str(mode_dic[coding_name]))
            shifty_par[coding_name][j] = str(shifty_par[coding_name][j])
    elif type(shifty_par[coding_name]) == int:
        shifty_par[coding_name] = str(shifty_par[coding_name])
    elif type(shifty_par[coding_name]) == str:
        pass
    return shifty_par

def par_processing(certain_par, shifty_par):
    par_op = {}
    for i in shifty_par.keys():
        if type(shifty_par[i]) == list and len(shifty_par[i]) > 1:
            par_op[i] = shifty_par[i]
        else:
            certain_par[i] = shifty_par[i]
    if len(par_op) > 0:
        return 'optuna', (certain_par, par_op)
    else:
        return 'assign', certain_par


def get_time():
    t_ = str(datetime.now())
    t_ = t_.replace(' ', '_')
    t_ = t_.replace(':', '-')
    return t_[2:t_.index('.')]


def get_step(certain_par):
    disting_num = {'1h': 1, '30m': 2, '15m': 4, '10m': 6}
    if certain_par['del_mode'] != 1:
        day_len = (certain_par['del_time'][1] - certain_par['del_time'][0]) if certain_par['del_mode'] == 2 else 24
        max_step = day_len * disting_num[certain_par['disting']]
        if certain_par['pre_len'] < max_step:
            certain_par['step_len'] = certain_par['pre_len']
        else:
            certain_par['step_len'] = max_step
    else:
        ZeroDivisionError("Please specify parameter \"step_len\"")
    return certain_par

def generate_random_list(length, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(length)]





