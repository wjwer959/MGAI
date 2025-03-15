import numpy as np
import pandas as pd
from utils.masking import normalization, inormalization


class DataAcquire:
    def __init__(self, path, data_name, disting, file_id: list, ratio, target_line, input_line, in_len,
                 del_mode=0, del_time=[4, 20]):
        """

        :param path: 数据集路径 示例：'./dataset'
        :param data_name: 数据集名称 示例：'HangZ'
        :param disting: 数据集分辨率 示例：'1h'
        :param file_id: 数据集文件序号 list 示例：[0, 1, 2]
        :param ratio：训练集、验证集、测试集比例 示例：[7, 1, 2]
        :param target_line: 预测列标题 list 可包含多列 示例：['GHI']
        :param input_line: 输入列标题 list 可包含多列 示例：['GHI']
        :param del_mode: 删除数据模式 0:全部保留；1:删除所有GHI为0的数据；2:删除制定范围内数据
        :param del_time: 当del_mode==2时， 保留数据范围[a, b)
        """

        def _err(del_mode, del_time):
            if del_mode not in [0, 1, 2]:
                raise ZeroDivisionError("The parameter \"del_mode\" must be in 0-2")
            if len(del_time) > 2:
                raise ZeroDivisionError("When \"del_mode\" is equal to 2, the length of \"del_time\" must be 2")

        def _filelist(path, data_name, disting, file_id):
            name_list = []
            for i in range(len(file_id)):
                file = '{path}\\{name}\\{name}_{disting}\\{name}_{id}.csv'.format(
                    path=path, name=data_name, disting=disting, id=str(file_id[i]))
                name_list.append(file)
            return name_list

        def _ratio_trans(ratio):
            add = sum(ratio)
            ratio_list = []
            for i in ratio:
                ratio_list.append(i/add)
            return ratio_list

        _err(del_mode, del_time)
        self.file_list = _filelist(path, data_name, disting, file_id)
        self.disting = disting
        self.ratio = _ratio_trans(ratio)
        self.input_line = input_line
        self.target_line = target_line
        self.del_mode = del_mode
        self.del_time = del_time
        self.in_len = in_len
    def get_data(self):
        def _mode1(dat, id):
            a = dat.loc[id, self.target_line]
            if a > 0.0:
                return True
            return False

        def _mode2(dat, id):
            if self.del_time[0] <= dat.loc[id, 'Hour'] < self.del_time[1]:
                return True
            return False

        def _mode_id(dat, id):
            if self.del_mode == 1:
                return _mode1(dat, id)
            elif self.del_mode == 2:
                return _mode2(dat, id)
            return ZeroDivisionError("Error")

        def _del(data: pd.DataFrame):
            save_id = []
            for i in range(data.shape[0]):
                if _mode_id(data, i):
                    save_id.append(i)
            return data.iloc[save_id, :]

        for i in range(len(self.file_list)):
            if i == 0:
                alldata = pd.read_csv(self.file_list[0])
            else:
                data_ = pd.read_csv(self.file_list[i])
                alldata = pd.concat((alldata, data_), axis=0, ignore_index=True)
        if self.del_mode != 0:
            return _del(alldata)
        return alldata


    def get_time_line(self):

        time_line = ['Month', 'Day', 'Hour']
        if self.disting[-1] == 'm':
            time_line.append('Minute')
        return time_line


    def data_segment(self, data, time_line):
        return np.array(data.loc[:, self.input_line]), np.array(data.loc[:, time_line]), np.array(data.loc[:, self.target_line])

    def set_segment1(self, whole_data: np.ndarray):
        le = len(whole_data)
        val_st = int(le*self.ratio[0])
        test_st = le - int(le*self.ratio[-1])
        return whole_data[:val_st], whole_data[val_st-self.in_len:test_st], whole_data[test_st-self.in_len:]

    def set_segment2(self, whole_data: np.ndarray):
        disting_num = {'1h': 1, '30m': 2, '15m': 4, '10m': 6}
        hour_num = (self.del_time[1] - self.del_time[0]) if self.del_mode == 2 else 24
        day_num = disting_num[self.disting] * hour_num
        num = len(whole_data) / day_num
        val_st = int(num*self.ratio[0]) * day_num
        test_st = int((num - num*self.ratio[-1])) * day_num
        if val_st / day_num == val_st // day_num and test_st / day_num == test_st // day_num:
            return whole_data[:val_st], whole_data[val_st-self.in_len:test_st], whole_data[test_st-self.in_len:]

    def set_segment(self, whole_data: np.ndarray):
        if self.del_mode == 1:
            return self.set_segment1(whole_data)
        else:
            return self.set_segment2(whole_data)

    def data(self):
        data_input, data_time, data_target = self.data_segment(self.get_data(), self.get_time_line())
        data_input, input_norm = normalization(data_input)
        data_target, target_norm = normalization(data_target)
        data_time, time_norm = normalization(data_time)
        input_train, input_val, input_test = self.set_segment(data_input)
        time_train, time_val, time_test = self.set_segment(data_time)
        target_train, target_val, target_test = self.set_segment(data_target)

        return (input_train, time_train, target_train), (input_val, time_val, target_val), (input_test, time_test, target_test), (input_norm, time_norm, target_norm)


