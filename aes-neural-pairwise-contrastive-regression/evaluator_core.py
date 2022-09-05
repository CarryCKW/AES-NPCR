# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-02-10 14:56:57
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-02-14 14:10:46
from utils import rescale_tointscore, get_logger
from metrics import *
import numpy as np
import torch.utils.data as Data
import torch


logger = get_logger("Evaluate stats")


class Evaluator():

    def __init__(self, prompt_id, use_char, out_dir, modelname, dev_x, dev_y, yy_dev,test_x, test_y,yy_test):
        self.use_char = use_char
        self.prompt_id = prompt_id
        self.dev_x, self.test_x = dev_x, test_x
        self.dev_y, self.test_y = dev_y, test_y
        self.dev_y_org = yy_dev
        self.test_y_org = yy_test
        self.out_dir = out_dir
        self.modelname = modelname
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test= [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

    def calc_correl(self, train_pred, dev_pred, test_pred):
        self.dev_pr = pearson(self.dev_y_org.squeeze(), dev_pred.squeeze())
        self.test_pr = pearson(self.test_y_org.squeeze(), test_pred.squeeze())

        self.dev_spr = spearman(self.dev_y_org, dev_pred)
        self.test_spr = spearman(self.test_y_org, test_pred)

    def calc_kappa(self, train_pred, dev_pred, test_pred, weight='quadratic'):
        self.dev_qwk = kappa(self.dev_y_org, dev_pred, weight)
        self.test_qwk = kappa(self.test_y_org, test_pred, weight)

    def calc_rmse(self, train_pred, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(self.dev_y_org, dev_pred)
        self.test_rmse = root_mean_square_error(self.test_y_org, test_pred)

    def evaluate(self, model, epoch, print_info=False):
        dev_pred_int = []
        test_pred_int = []
        print("------")
        for i in range(len(self.dev_x)):
            dev_x0 = [j[0] for j in self.dev_x[i]]
            dev_x1 = [j[1] for j in self.dev_x[i]]
            dev_pred = []
            for j in range(len(dev_x0)):
                dev_x00 = torch.LongTensor(dev_x0[j])
                dev_x10 = torch.LongTensor(dev_x1[j])
                dev_pred0 = model((dev_x00.unsqueeze(dim=0)).cuda(), (dev_x10.unsqueeze(dim=0)).cuda()).squeeze()
                dev_pred0 = dev_pred0.cpu()
                dev_pred.append(dev_pred0)
            dev_pred = torch.Tensor(dev_pred)
            dev_predint = rescale_tointscore(dev_pred,self.prompt_id)
            dev_predd = []
            for m in range(len(self.dev_y[i])):
                dev_predd.append(dev_predint[m] + self.dev_y[i][m])
            dev_predd = np.array(dev_predd)
            dev_int = np.mean(dev_predd)
            dev_int_round = np.around([dev_int]).astype(int)  # add
            # dev_pred_int.append([dev_int])
            dev_pred_int.append(dev_int_round)
        dev_predd_int = np.array(dev_pred_int)
        print("------")
        for i in range(len(self.test_x)):
            test_x0 = [j[0] for j in self.test_x[i]]
            test_x1 = [j[1] for j in self.test_x[i]]
            test_pred = []
            for j in range(len(test_x0)):
                test_x00 = torch.LongTensor(test_x0[j])
                test_x10 = torch.LongTensor(test_x1[j])
                test_pred0 = model((test_x00.unsqueeze(dim=0)).cuda(), (test_x10.unsqueeze(dim=0)).cuda()).squeeze()
                test_pred0 = test_pred0.cpu()
                test_pred.append(test_pred0)
            test_pred = torch.Tensor(test_pred)
            test_predint = rescale_tointscore(test_pred, self.prompt_id)
            test_predd = []
            for m in range(len(self.test_y[i])):
                test_predd.append(test_predint[m] + self.test_y[i][m])
            test_predd = np.array(test_predd)
            test_int = np.mean(test_predd)
            test_int_round = np.around([test_int]).astype(int)
            # test_pred_int.append([test_int])
            test_pred_int.append(test_int_round)
        test_predd_int = np.array(test_pred_int)

        self.calc_correl(None, dev_predd_int, test_predd_int)
        self.calc_kappa(None, dev_predd_int, test_predd_int)
        self.calc_rmse(None, dev_predd_int, test_predd_int)

        if self.dev_qwk > self.best_dev[0]:
            self.best_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.dev_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
            self.best_dev_epoch = epoch
            torch.save(model, self.out_dir + '/dev' + self.modelname)

        if self.test_qwk > self.best_test[0]:
            self.best_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
            self.test_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.best_test_epoch = epoch
            torch.save(model, self.out_dir + '/test' + self.modelname)

        if print_info:
            self.print_info()

    def print_info(self):
        logger.info('[DEV]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse, self.best_dev_epoch,
            self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))

        logger.info('[TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_test_epoch,
            self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

        logger.info(
            '[DEV-TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
                self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_dev_epoch,
                self.dev_test[0], self.dev_test[1], self.dev_test[2], self.dev_test[3]))

        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
        logger.info('[BEST DEV for TEST] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
            self.dev_test[0], self.dev_test[1], self.dev_test[2], self.dev_test[3]))


class Evaluator_opti():
    def __init__(self, prompt_id, use_char, out_dir, modelname, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal, example_size):
        self.use_char = use_char
        self.prompt_id = prompt_id

        self.features_dev = features_dev
        self.dev_y_example = dev_y_example
        self.dev_y_goal = dev_y_goal

        self.features_test = features_test
        self.test_y_example = test_y_example
        self.test_y_goal = test_y_goal

        self.example_size = example_size

        self.out_dir = out_dir
        self.modelname = modelname
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test= [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

        self.dev_loader, self.test_loader = self.init_data_loader()

    def calc_correl(self, train_pred, dev_pred, test_pred):
        self.dev_pr = pearson(self.dev_y_goal.squeeze(), dev_pred.squeeze())
        self.test_pr = pearson(self.test_y_goal.squeeze(), test_pred.squeeze())

        self.dev_spr = spearman(self.dev_y_goal, dev_pred)
        self.test_spr = spearman(self.test_y_goal, test_pred)

    def calc_kappa(self, train_pred, dev_pred, test_pred, weight='quadratic'):
        self.dev_qwk = kappa(self.dev_y_goal, dev_pred, weight)
        self.test_qwk = kappa(self.test_y_goal, test_pred, weight)

    def calc_rmse(self, train_pred, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(self.dev_y_goal, dev_pred)
        self.test_rmse = root_mean_square_error(self.test_y_goal, test_pred)

    def print_info(self):
        logger.info('[DEV]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse, self.best_dev_epoch,
            self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))

        logger.info('[TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_test_epoch,
            self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

        logger.info(
            '[DEV-TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
                self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_dev_epoch,
                self.dev_test[0], self.dev_test[1], self.dev_test[2], self.dev_test[3]))

        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
        logger.info('[BEST DEV for TEST] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
        self.dev_test[0], self.dev_test[1], self.dev_test[2], self.dev_test[3]))

    def evaluate(self, model, epoch, print_info=False):
        dev_pred_int = []
        for step, (batch_dev_x0, batch_dev_x1, batch_example_s) in enumerate(self.dev_loader):
            dev_y_pred = model(batch_dev_x0.cuda(), batch_dev_x1.cuda())
            dev_y_pred = dev_y_pred.cpu()
            dev_y_pred_rescale = torch.Tensor(rescale_tointscore(dev_y_pred, self.prompt_id)) # [examlpe_size, 1]
            dev_pred_group = dev_y_pred_rescale + batch_example_s
            dev_pred = torch.mean(dev_pred_group, dim=0).numpy()
            dev_pred_i = np.around(dev_pred).astype(int)
            dev_pred_int.append(dev_pred_i)
        dev_pred_int = np.array(dev_pred_int)

        test_pred_int = []
        for step, (batch_test_x0, batch_test_x1, batch_example_s) in enumerate(self.test_loader):
            test_y_pred = model(batch_test_x0.cuda(), batch_test_x1.cuda())
            test_y_pred = test_y_pred.cpu()
            test_y_pred_rescale = torch.Tensor(rescale_tointscore(test_y_pred, self.prompt_id)) # [examlpe_size, 1]
            test_pred_group = test_y_pred_rescale + batch_example_s
            test_pred = torch.mean(test_pred_group, dim=0).numpy()
            test_pred_i = np.around(test_pred).astype(int)
            test_pred_int.append(test_pred_i)
        test_pred_int = np.array(test_pred_int)

        self.calc_correl(None, dev_pred_int, test_pred_int)
        self.calc_kappa(None, dev_pred_int, test_pred_int)
        self.calc_rmse(None, dev_pred_int, test_pred_int)

        if self.dev_qwk > self.best_dev[0]:
            self.best_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.dev_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
            self.best_dev_epoch = epoch
            torch.save(model, self.out_dir + '/dev' + self.modelname)

        if self.test_qwk > self.best_test[0]:
            self.best_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
            self.test_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.best_test_epoch = epoch
            torch.save(model, self.out_dir + '/test' + self.modelname)

        if print_info:
            self.print_info()

    def init_data_loader(self):
        dev_x0 = [j[0] for j in self.features_dev]
        dev_x1 = [j[1] for j in self.features_dev]
        dev_x0 = torch.LongTensor(dev_x0)
        dev_x1 = torch.LongTensor(dev_x1)
        dev_example_s = torch.Tensor(self.dev_y_example)

        test_x0 = [j[0] for j in self.features_test]
        test_x1 = [j[1] for j in self.features_test]
        test_x0 = torch.LongTensor(test_x0)
        test_x1 = torch.LongTensor(test_x1)
        test_example_s = torch.Tensor(self.test_y_example)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_example_s)
        test_dataset = Data.TensorDataset(test_x0, test_x1, test_example_s)

        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=self.example_size, drop_last=False
            ),
            num_workers=2
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=test_dataset), batch_size=self.example_size, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader, test_loader
