# -*- coding: utf-8 -*-
# file: polarity_infer.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from models.lcf_bert import LCF_BERT
from models.lcf_glove import LCF_GLOVE
from pytorch_transformers import BertModel
from data_utils import Tokenizer4Bert
import argparse


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def prepare_data(text_left, aspect, text_right, tokenizer):
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    
    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)            
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices


def get_parameters():
    # Keep the follow parameters same as training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--dataset', default='phone', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--use_single_bert', default=False, type=bool,
                        help='use the same bert modeling for global context and local context to reduce memory requirement')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--SRD', default=5, type=int)
    parser.add_argument('--local_context_focus', default='cdm', type=str)
    parser.add_argument('--seed', default=2019, type=str)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    model_classes = {
        'lcf_bert': LCF_BERT,
        'lcf_glove': LCF_GLOVE
    }
    # set your trained models here
    state_dict_paths = {
        'lcf_bert': 'state_dict/lcf_bert_car_acc96.84'
    }

    opt = get_parameters()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = model_classes[opt.model_name](bert, opt).to(opt.device)
    
    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(state_dict_paths[opt.model_name]))
    model.eval()
    torch.autograd.set_grad_enabled(False)

    # input: This little place has a cute interior decor and affordable city prices.
    # text_left = This little place has a cute 
    # aspect = interior decor
    # text_right = and affordable city prices.
    
    text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data('This little place has a cute', 'interior decor', 'and affordable city prices.', tokenizer)
    
    
    text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
    bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
    text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
    aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
    if 'lcf' in opt.model_name:
        inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices]
    outputs = model(inputs)
    t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
    print('t_probs = ', t_probs)
    print('aspect sentiment = ', t_probs.argmax(axis=-1) - 1)
