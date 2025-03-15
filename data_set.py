from torch.utils.data import Dataset
import numpy as np
import torch


class DataExtract(Dataset):
    def __init__(self, data, in_len, pre_len, label_len, step):
        self.input, self.time, self.target = data
        self.alllen = self.input.shape[0]
        if self.alllen == self.time.shape[0] and self.alllen == self.target.shape[0]:
            self.in_len = in_len
            self.pre_len = pre_len
            self.label_len = label_len
            self.step = step
        else:
            raise ZeroDivisionError("Data shape error")

    def __getitem__(self, item):
        item = item * self.step
        in_end = item + self.in_len
        out_begin = in_end - self.label_len
        out_end = in_end + self.pre_len
        input_ = self.input[item:in_end, :]
        time_in = self.time[item:in_end, :]
        target_ = self.target[out_begin:out_end, :]
        time_out = self.time[out_begin:out_end, :]

        return input_, time_in, target_, time_out

    def __len__(self):
        return (self.alllen - self.in_len - self.pre_len) // self.step + 1




