import optuna
import scipy.io as sio
import pandas as pd
import time
import numpy as np
import os
from utils.masking import model_type, par_processing, purification_par, get_step

from exp_main import exp_fit
from utils.masking import get_time


class OptunaPar:
    def __init__(self, par, opt_par):
        self.best_fit = None
        self.fit = None
        self.par = par
        self.opt_par = opt_par
        self.n_trial = par['n_trial']
        # self.model_type = model_type(par['model_name'])        # 0:former  1:rnn


    def fit_(self, trial):
        '''if self.par['d_model_1'] >= self.par['in_len'] or self.par['label_len'] > self.par['in_len']:
            raise optuna.exceptions.TrialPruned()'''
        loss_best = 100.0
        flag_max = 5
        flag = 0

        for epoch in range(2000):
            self.fit.train(epoch)
            loss = self.fit.val()
            if epoch >= 10:
                if loss < loss_best:
                    loss_best = loss
                    flag = 0
                else:
                    flag += 1
                if flag >= flag_max:
                    break
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return loss_best

    def objective(self, trial):
        def par_step(par_name):
            if len(self.opt_par[par_name]) != 3:
                return False
            else:
                for i in self.opt_par[par_name]:
                    if type(i) != int and type(i) != float:
                        return False
                return True

        def par_species(trial, ty):
            if ty == float:
                return trial.suggest_float
            elif ty == int:
                return trial.suggest_int
            elif ty == str:
                return trial.suggest_categorical

        for par_ in self.opt_par.keys():
            if par_step(par_):
                self.par[par_] = par_species(trial, type(self.opt_par[par_][-1]))(par_, self.opt_par[par_][0],
                                                                                  self.opt_par[par_][1],
                                                                                  step=self.opt_par[par_][2])
            else:
                if type(self.opt_par[par_][-1]) != str and self.opt_par[par_] == 2:
                    self.par[par_] = par_species(trial, type(self.opt_par[par_][-1]))(par_, self.opt_par[par_][0],
                                                                                      self.opt_par[par_][1])
                else:
                    self.par[par_] = trial.suggest_categorical(par_, self.opt_par[par_])
        self.par['d_model'] = self.par['d_model_1'] * self.par['n_heads']
        if 'local_coding_mode' in self.par.keys():
            self.par['coding_mode'] = self.par['local_coding_mode'] + self.par['global_coding_mode']
        print('par: ', self.par)
        self.fit = exp_fit(self.par)
        return self.fit_(trial)

    def run(self):
        best_par = self.par
        study = optuna.create_study(direction='minimize')      #   , storage="sqlite:///db.sqlite3")
        study.optimize(self.objective, n_trials=self.par['n_trial'])
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        path = './res_/' + self.par['data_name'] + '/' + self.par['disting'] + '/' + self.par['model_name'] + '/op_' + \
               self.par['model_name'] + '_' + self.par['disting'] + '_' + str(self.par['pre_len'])
        for key, value in trial.params.items():
            best_par[key] = value
            print("    {}: {}".format(key, value))
        sio.savemat(path + '.mat', best_par)
        #  self.best_fit.test()




class AssignPar:
    def __init__(self, par_dic, i=0):
        self.i = i
        self.par = par_dic
        self.n_trial = par_dic['n_trial']
        # self.model_type = model_type(par['model_name'])        # 0:former  1:rnn


    def run(self):
        if 'local_coding_mode' in self.par.keys():
            self.par['coding_mode'] = self.par['local_coding_mode'] + self.par['global_coding_mode']
        if 'd_model_1' in self.par.keys():
            self.par['d_model'] = self.par['d_model_1'] * self.par['n_heads']
        fit = exp_fit(self.par)
        loss_best = 100.0
        flag_max = 5
        flag = 0
        LOSS_FILE = []
        for epoch in range(2000):
            fit.train(epoch)
            loss = fit.val()
            LOSS_FILE.append(loss)
            # fit.test()
            if epoch >= 10:
                if loss < loss_best:
                    loss_best = loss
                    best_epoch = epoch
                    best_fit = fit
                    flag = 0
                else:
                    flag += 1
                if flag >= flag_max:
                    break
        print('best_epoch: {}, best_loss: {}'.format(best_epoch, loss_best))
        df = pd.DataFrame(LOSS_FILE, columns=['Value'])
        df.to_csv('./' + str(int(time.time())) + '.csv', index=False, encoding='utf-8')
        best_fit.test(self.i)


def f_run(certain_par, shifty_par, i=0):
    certain_par = get_step(certain_par) if certain_par['step_len'] is None else certain_par
    run_mode, par = par_processing(certain_par, purification_par(certain_par['model_name'], shifty_par))
    if run_mode == 'optuna':
        par_cer, par_op = par
        print('Hyperparameters optimized by optuna is ', par_op)
        OptunaPar(par=par_cer, opt_par=par_op).run()
    elif run_mode == 'assign':
        print('All hyperparameters are specified')
        AssignPar(par, i).run()












