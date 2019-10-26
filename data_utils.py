# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# modified: yangheng<yangheng@m.scnu.edu.cn>

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
import argparse
import json

def parse_experiments(path):


    configs = []

    opt = argparse.ArgumentParser()
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default=config['model_name'], type=str)
        parser.add_argument('--dataset', default=config['dataset'], type=str, help='twitter, restaurant, laptop')
        parser.add_argument('--use_single_bert', default=config['use_single_bert'], type=bool,
                            help='use the same bert layer for global context and local context to reduce memory requirement')
        parser.add_argument('--optimizer', default=config['optimizer'], type=str)
        parser.add_argument('--initializer', default=config['initializer'], type=str)
        parser.add_argument('--learning_rate', default=config['learning_rate'], type=float)
        parser.add_argument('--dropout', default=config['dropout'], type=float)
        parser.add_argument('--l2reg', default=config['l2reg'], type=float)
        parser.add_argument('--num_epoch', default=config['num_epoch'], type=int)
        parser.add_argument('--batch_size', default=config['batch_size'], type=int)
        parser.add_argument('--log_step', default=config['log_step'], type=int)
        parser.add_argument('--logdir', default=config['logdir'], type=str)
        parser.add_argument('--bert_dim', default=config['bert_dim'], type=int)
        parser.add_argument('--pretrained_bert_name', default=config['pretrained_bert_name'], type=str)
        parser.add_argument('--max_seq_len', default=config['max_seq_len'], type=int)
        parser.add_argument('--polarities_dim', default=config['polarities_dim'], type=int)
        parser.add_argument('--hops', default=config['hops'], type=int)
        parser.add_argument('--SRD', default=config['SRD'], type=int)
        parser.add_argument('--local_context_focus', default=config['local_context_focus'], type=str)
        configs.append(parser.parse_args())
    return configs

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[len(tokens)-300:len(tokens)], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.840B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


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


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        print('buliding word indices...')
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1

            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            text_left = ' '.join(text_left.split(' ')[int(-(tokenizer.max_seq_len-aspect_len)/2-5):])
            text_right = ' '.join(text_right.split(' ')[:int((tokenizer.max_seq_len-aspect_len)/2-5)])
            text_raw = text_left +' '+ aspect +' '+ text_right
            text_raw_indices = tokenizer.text_to_sequence(text_raw)

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")


            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'polarity': polarity,
            }

            all_data.append(data)

        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

