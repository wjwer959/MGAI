import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, par):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=len(par['input_line']), hidden_size=par['d_model'], num_layers=par['num_layers'],
                            bias=True, batch_first=True)
        self.linear1_1 = nn.Linear(par['d_model'], par['d_model'] // 2)
        self.linear1_2 = nn.Linear(par['d_model'] // 2, 1)
        self.linear2_1 = nn.Linear(par['in_len'], par['in_len'] // 2)
        self.linear2_2 = nn.Linear(par['in_len'] // 2, par['pre_len'])

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        data = x.float()
        data, (_, _) = self.lstm(data)
        data = self.linear1_1(data)
        data = F.relu(data)
        data = self.linear1_2(data)
        data = data.transpose(1, 2)
        data = self.linear2_1(data)
        data = F.relu(data)
        data = self.linear2_2(data)
        data = data.transpose(1, 2)
        return data
