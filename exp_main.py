import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import os
import scipy.io as sio

from models import Informer, Transformer, LSTNet, LSTM
from models import DeepAR0
import Segformer
from data_set import DataExtract
from utils.masking import model_type, get_time
from data_dispose import DataAcquire
from utils.masking import inormalization


def exp_fit(par_dic):

    mod_type = model_type(par_dic['model_name'])
    if mod_type == 0:
        return ExpFit(par_dic)
    elif mod_type == 1:
        return ExpFitdeepar(par_dic)



class ExpFit:
    def __init__(self, par_dic):

        def _optimizer_allot(optimizer_name, lr):
            if optimizer_name == 'Adam':
                return torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer_name == 'SGD':
                return torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer_name == 'RMSprop':
                return torch.optim.RMSprop(self.model.parameters(), lr=lr)
            else:
                raise ZeroDivisionError('The optimizer input is incorrect')

        def _loss_allot(loss_name):
            if loss_name == 'MSE':
                return nn.MSELoss()
            elif loss_name == 'MAE':
                return nn.L1Loss()
            else:
                raise ZeroDivisionError('The Loss input is incorrect')

        model_list = {
            # 'autoformer':   Autoformer.Model,
            'informer': Informer.Model,
            'transformer': Transformer.Model,
            'segformer': Segformer.Model,
            'lstnet': LSTNet.Model,
            'lstm': LSTM.Model

        }
        self.par = par_dic
        self.device = torch.device(par_dic['device'])
        self.c_in = len(par_dic['input_line'])
        self.c_out = len(par_dic['target_line'])
        # self.model_type = model_type(par_dic['model_name'])

        self.model = model_list[par_dic['model_name']](par_dic).to(self.device)
        self.criterion = _loss_allot(par_dic['loss'])
        self.optimizer = _optimizer_allot(par_dic['optimizer'], par_dic['lr'])
        data_train, data_val, data_test, (_, _, self.norm) = DataAcquire(path=par_dic['path'],
                                                                         data_name=par_dic['data_name'],
                                                                         disting=par_dic['disting'],
                                                                         file_id=par_dic['file_id'],
                                                                         ratio=par_dic['ratio'],
                                                                         target_line=par_dic['target_line'],
                                                                         input_line=par_dic['input_line'],
                                                                         in_len=par_dic['in_len'],
                                                                         del_mode=par_dic['del_mode'],
                                                                         del_time=par_dic['del_time']).data()
        self.train_loader = self.get_loder(data_train, 'train')
        self.val_loader = self.get_loder(data_val, 'val')
        self.test_loader = self.get_loder(data_test, 'test')

    def get_loder(self, data, course):
        drop_shuffle = False if course != 'train' else True
        step = self.par['step_len'] if course != 'test' else self.par['pre_len']
        set_ = DataExtract(data, self.par['in_len'], self.par['pre_len'], self.par['label_len'], step)
        return DataLoader(set_, batch_size=self.par['batch_size'], drop_last=drop_shuffle, shuffle=drop_shuffle)

    def predict(self, batch_x, batch_x_time, batch_y, batch_y_time):
        # input_encoder = batch_x
        input_decoder = torch.zeros_like(batch_y[:, -self.par['pre_len']:, :])
        input_decoder = torch.cat([batch_y[:, :self.par['label_len'], :], input_decoder], dim=1).to(self.device)

        output = self.model(batch_x, batch_x_time, input_decoder, batch_y_time)
        if self.par['out_att']:
            return output
        else:
            return output, None

    def restore(self, data_):
        if data_.shape[1] == self.par['label_len'] + self.par['pre_len']:
            return data_[:, -self.par['pre_len']:, :]
        else:
            print(data_.shape)
            print(self.par['label_len'], self.par['pre_len'])
            raise ZeroDivisionError('The shape of the data entered is wrong')

    def train(self, epoch_num):
        self.model.train()
        loss_epoch = []
        t1 = time.time()
        for idx, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                batch_y_time.float()
            batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                batch_y.to(self.device), batch_y_time.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.predict(batch_x, batch_x_time, batch_y, batch_y_time)
            batch_y, output = self.restore(batch_y), self.restore(output)
            loss = self.criterion(output, batch_y)
            loss_epoch.append(loss.item())
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('Train epoch:{0}, Loss:{1}, time:{2}'.format(epoch_num, loss_, t2-t1))
        # return loss_

    def val(self):
        self.model.eval()
        loss_epoch = []
        t = time.time()
        with torch.no_grad():
            for idx, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.val_loader, 0):
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                    batch_y_time.float()
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                    batch_y.to(self.device), batch_y_time.to(self.device)
                output, _ = self.predict(batch_x, batch_x_time, batch_y, batch_y_time)
                batch_y, output = self.restore(batch_y), self.restore(output)

                loss = self.criterion(output, batch_y)
                loss_epoch.append(loss.item())
        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('    Validation, Loss:{0}, time:{1}'.format(loss_, time.time()-t))
        return loss_
        # return loss_

    def test(self, i):

        self.model.eval()
        loss_epoch = []
        att_set = []
        # tar_data, pre_data = [], []
        t = time.time()
        with torch.no_grad():
            for idx, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.test_loader, 0):
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                    batch_y_time.float()
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                    batch_y.to(self.device), batch_y_time.to(self.device)
                output, att = self.predict(batch_x, batch_x_time, batch_y, batch_y_time)
                batch_y, output = self.restore(batch_y), self.restore(output)
                loss = self.criterion(output, batch_y)
                loss_epoch.append(loss.item())
                if idx == 0:
                    tar_data, pre_data = np.array(batch_y.cpu()), np.array(output.cpu())
                else:
                    tar_data, pre_data = np.concatenate((tar_data, batch_y.cpu()), 0), np.concatenate((pre_data, output.cpu()), 0)
                if att is not None:
                    for i in range(len(att)):
                        att[i] = att[i].cpu().reshape(-1, self.par['in_len'], self.par['in_len'])
                    att_set = np.array(att[0])
        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('    Test, Loss:{0}, time:{1}'.format(loss_, time.time() - t))
        path = self.par['data_name'] + '/' + self.par['disting'] + '/' + self.par['model_name'] + '/' + self.par['model_name'] + '_' + self.par['disting'] + '_' + str(self.par['pre_len']) + str(i) # + '_' + get_time()
        if not os.path.exists(path):
            os.mkdir(path)
        #torch.save(self.model.state_dict(), path+'/1model.pth')

        #np.save(path + '/5target.npy', np.array(tar_data))
        #np.save(path + '/5output.npy', np.array(pre_data))
        #np.save(path + '/5norm.npy', self.norm)
        if att is not None:
            np.save(path + '/att.npy', att_set)
        # sio.savemat(path + '/result.mat', result)
        #sio.savemat(path + '/1par.mat', self.par)











class ExpFitdeepar:
    def __init__(self, par_dic):

        def _optimizer_allot(optimizer_name, lr):
            if optimizer_name == 'Adam':
                return torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer_name == 'SGD':
                return torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer_name == 'RMSprop':
                return torch.optim.RMSprop(self.model.parameters(), lr=lr)
            else:
                raise ZeroDivisionError('The optimizer input is incorrect')


        self.par = par_dic
        self.device = torch.device(par_dic['device'])
        self.c_in = len(par_dic['input_line'])
        self.c_out = len(par_dic['target_line'])
        # self.model_type = model_type(par_dic['model_name'])
        self.criterion = DeepAR0.loss
        self.criterion_ = nn.MSELoss()
        self.model = DeepAR0.Model(par_dic).to(self.device)
        self.optimizer = _optimizer_allot(par_dic['optimizer'], par_dic['lr'])
        data_train, data_val, data_test, (_, _, self.norm) = DataAcquire(path=par_dic['path'],
                                                                         data_name=par_dic['data_name'],
                                                                         disting=par_dic['disting'],
                                                                         file_id=par_dic['file_id'],
                                                                         ratio=par_dic['ratio'],
                                                                         target_line=par_dic['target_line'],
                                                                         input_line=par_dic['input_line'],
                                                                         in_len=par_dic['in_len'],
                                                                         del_mode=par_dic['del_mode'],
                                                                         del_time=par_dic['del_time']).data()
        self.train_loader = self.get_loder(data_train, 'train')
        self.val_loader = self.get_loder(data_val, 'val')
        self.test_loader = self.get_loder(data_test, 'test')

    def get_loder(self, data, course):
        drop_shuffle = False if course != 'train' else True
        step = self.par['step_len'] if course != 'test' else self.par['pre_len']
        set_ = DataExtract(data, self.par['in_len'], self.par['pre_len'], self.par['label_len'], step)
        return DataLoader(set_, batch_size=self.par['batch_size'], drop_last=drop_shuffle, shuffle=drop_shuffle)

    def restore(self, data_):
        if data_.shape[1] == self.par['label_len'] + self.par['pre_len']:
            return data_[:, -self.par['pre_len']:, :]
        else:
            print(data_.shape)
            print(self.par['label_len'], self.par['pre_len'])
            raise ZeroDivisionError('The shape of the data entered is wrong')

    def train(self, epoch_num):
        self.model.train()
        loss_epoch = []

        t1 = time.time()
        for idx, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.train_loader):
            batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                batch_y_time.float()
            batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                batch_y.to(self.device), batch_y_time.to(self.device)
            self.optimizer.zero_grad()
            mu, sigma = self.model(batch_x)
            loss = self.criterion(mu, sigma, batch_y) + self.criterion_(mu, batch_y)
            loss.backward()
            self.optimizer.step()
            loss_epoch.append(loss.item())
        t2 = time.time()
        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('Train epoch:{0}, Loss:{1}, time:{2}'.format(epoch_num, loss_, t2-t1))
        # return loss_

    def val(self):
        self.model.eval()
        loss_epoch = []
        t = time.time()
        with torch.no_grad():
            for idx, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.val_loader, 0):
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                    batch_y_time.float()
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                    batch_y.to(self.device), batch_y_time.to(self.device)
                mu, sigma = self.model(batch_x)
                loss = self.criterion(mu, sigma, batch_y) + self.criterion_(mu, batch_y)
                loss_epoch.append(loss.item())
        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('    Validation, Loss:{0}, time:{1}'.format(loss_, time.time()-t))
        return loss_
        # return loss_

    def test(self):

        self.model.eval()
        loss_epoch = []
        att_set = []
        # tar_data, pre_data = [], []
        t = time.time()
        with torch.no_grad():
            for idx_, (batch_x, batch_x_time, batch_y, batch_y_time) in enumerate(self.test_loader, 0):
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.float(), batch_x_time.float(), batch_y.float(), \
                    batch_y_time.float()
                batch_x, batch_x_time, batch_y, batch_y_time = batch_x.to(self.device), batch_x_time.to(self.device), \
                    batch_y.to(self.device), batch_y_time.to(self.device)
                mu, sigma = self.model(batch_x)
                loss = self.criterion(mu, sigma, batch_y) + self.criterion_(mu, batch_y)
                loss_epoch.append(loss.item())
                if idx_ == 0:
                    tar_data, pre_data = np.array(batch_y.cpu()), np.array(mu.cpu())
                else:
                    tar_data, pre_data = np.concatenate((tar_data, batch_y.cpu()), 0), np.concatenate((pre_data, mu.cpu()), 0)

        loss_ = sum(loss_epoch) / len(loss_epoch)
        print('    Test, Loss:{0}, time:{1}'.format(loss_, time.time() - t))
        path = self.par['data_name'] + '/' + self.par['disting'] + '/' + self.par['model_name'] + '/' + self.par[
            'model_name'] + '_' + self.par['disting'] + '_' + str(self.par['pre_len']) + str(i)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.model.state_dict(), path+'/model.pth')
        np.save(path + '/target.npy', np.array(tar_data).reshape(-1, self.c_out))
        np.save(path + '/output.npy', np.array(pre_data).reshape(-1, self.c_out))
        np.save(path + '/norm.npy', self.norm)
        # sio.savemat(path + '/result.mat', result)
        sio.savemat(path + '/par.mat', self.par)













