# -*- coding: utf-8 -*-
# file: batch_train.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


from pytorch_transformers import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import math
import os
import numpy, random

from data_utils import Tokenizer4Bert, ABSADataset, parse_experiments

from models import LCF_BERT
from models.bert_spc import BERT_SPC

import logging,sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)

        self.model = opt.model_class(bert, opt).to(opt.device)

        trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logging.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=opt.device.index)))

        self._log_write_args()

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logging.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.opt):
            logging.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params (with unfreezed bert)
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, max_test_acc_overall=0):

        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            logging.info('>' * 100)
            logging.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_{1}_acc{2}'.format(self.opt.model_name, self.opt.dataset,
                                                                      round(test_acc * 100, 2))
                            # torch.save(self.model.state_dict(), path)
                            logging.info('>> saved: ' + path)
                        logging.info('max_acc:{}  f1:{}'.format(round(test_acc * 100, 2),round(f1 * 100, 2)))
                    if f1 > max_f1:
                        max_f1 = f1
                    logging.info('loss: {:.4f}, acc: {:.2f}, test_acc: {:.2f}, f1: {:.2f}'.format(loss.item(),train_acc*100,test_acc*100, f1*100))
        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            logging.info('repeat: {}'.format(i))
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall=max_test_acc_overall)
            # logging.info('max_test_acc: {0}     max_f1: {1}'.format(max_test_acc, max_f1))

            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            logging.info('#' * 100)

        logging.info("max_test_acc_overall:{}".format(max_test_acc_overall * 100))
        logging.info("max_f1_overall:{}".format(max_f1_overall * 100))
        return max_test_acc_overall * 100, max_f1_overall * 100


def single_train(opt):


    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_spc': BERT_SPC,
        'lcf_bert': LCF_BERT,
        'lca_net': LCF_BERT,
    }

    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
            },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }

    }

    input_colses = {
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'lca_net': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    return ins.run()  # _reset_params in every repeat


def multi_train(config, n):
    import copy
    mean_test_acc_overall = 0
    mean_f1_overall = 0
    temp_test_acc_overall = 0
    temp_f1_overall = 0
    max_acc_overall = 0
    max_f1_overall = 0
    scores = []
    for t in range(n):
        logging.info('{} - {} - {} - No.{} in {}'.format(config.model_name, config.dataset, config.local_context_focus, t + 1,  n))
        config.device = 'cuda:' + str(gpu)
        config.seed = t
        test_acc_overall, f1_overall = single_train(copy.deepcopy(config))
        scores.append([test_acc_overall, f1_overall])
        temp_test_acc_overall += test_acc_overall
        temp_f1_overall += f1_overall
        logging.info("#"*100)
        for i in range(len(scores)):
            if scores[i][0] > max_acc_overall:
                max_acc_overall = scores[i][0]
                max_f1_overall = scores[i][1]
            logging.info("{} test_acc_overall: {}  f1_overall:{}".format(i+1,round(scores[i][0], 2), round(scores[i][1], 2)))
        mean_test_acc_overall = temp_test_acc_overall / (t + 1)
        mean_f1_overall = temp_f1_overall / (t + 1)
        logging.info('max_acc_overall:{}  f1_overall:{}'.format(round(max_acc_overall, 2),round(max_f1_overall, 2)))
        logging.info("mean_acc_overall:{}  mean_f1_overall:{}".format(round(mean_test_acc_overall, 2), round(mean_f1_overall, 2)))
        logging.info("#" * 100)
    return mean_test_acc_overall, mean_f1_overall

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='experiments.json', type=str)
    parser.add_argument('--config', default='experiments.json', type=str)
    parser.add_argument('--repeat', default=5, type=int)
    args = parser.parse_args()
    path = args.config
    configs = parse_experiments(args.config)

    from lib.Pytorch_GPUManager import GPUManager

    GM = GPUManager()
    gpu = GM.auto_choice()

    for config in configs:
        config.device = 'cuda:'+str(gpu)
        multi_train(config=config, n=args.repeat)

