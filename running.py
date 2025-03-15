from stip_par import *

if __name__ == '__main__':
    def out_par(data_name, model_name, disting='1h', pre_len=16):
        certain_par = {
            'path': './dataset',
            'data_name': data_name,
            'file_id': [0, 1, 2, 3, 4],
            'ratio': [7, 1, 2],
            'target_line': ['GHI'],
            'input_line': ['GHI'],
            'del_mode': 2,  # 删除数据模式 0:全部保留；1:删除所有GHI为0的数据；2:删除制定范围内数据
            'del_time': [4, 20],
            'device': 'cuda',
            'disting': disting,
            'pre_len': pre_len,
            'step_len': None,

            'model_name': model_name,
            'n_trial': 100,
            'out_att': False,
        }

        shifty_par = {
            'optimizer': 'Adam',
            'loss': 'MSE',
            'lr': 0.0005,
            'dropout': 0.1,
            'activation': 'relu',

            'batch_size': 64,

            'in_len': 384,
            'd_model_1': [8, 64],
            'factor': 10,

            'label_len': 192,
            'n_heads': [2, 8],
            'e_layers': [1, 6],
            'd_layers': [1, 6],
            'local_coding_mode': 1,
            'global_coding_mode': 3,

            'num_layers': 6,

            'timesteps_': 500
        }
        return certain_par, shifty_par
    certain_par, shifty_par = out_par('HangZ', 'transformer', '10m', 384)
    f_run(certain_par, shifty_par)


