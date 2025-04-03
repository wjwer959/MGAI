from stip_par import *

if __name__ == '__main__':
    def out_par(data_name, disting='1h', pre_len=16):
        certain_par = {
            'path': './dataset',
            'data_name': data_name,
            'file_id': [0, 1, 2, 3, 4],
            'ratio': [7, 1, 2],
            'target_line': ['GHI'],
            'input_line': ['GHI'],
            'del_mode': 2,
            'del_time': [4, 20],
            'device': 'cuda',
            'disting': disting,
            'pre_len': pre_len,
            'step_len': None,

            'swo_lb': 1,
            'swo_ub': 10,   # 必须调整
            't_max': 200,
            'epsilon': 0.00001,

            'model_name': 'mgai',
            'n_trial': 100,
            'out_att': False,
        }

        shifty_par = {
            'optimizer': 'Adam',
            'loss': 'MSE',
            'lr': 0.0001,
            'dropout': 0.1,
            'activation': 'relu',
            'batch_size': 64,

            'in_len': 384,   # 必须调整
            'd_model_1': 32,
            'factor_init': 6,
            'label_len': 96,
            'n_heads': 8,
            'e_layers': 6,
            'd_layers': 2,
            'local_coding_mode': 1,
            'global_coding_mode': 3,
            'max_local_pos': 48,

            'grain_num': 3,
            'grain_layer': 2,
        }
        return certain_par, shifty_par
    certain_par, shifty_par = out_par('HangZ', '10m', 384)
    f_run(certain_par, shifty_par)

