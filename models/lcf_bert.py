# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention

class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class LCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert) if not opt.use_single_bert else self.bert4global
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear2 = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear3 = nn.Linear(opt.bert_dim * 3, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    # create the mask matrix for generating local context features
    def get_mask_matrix(self, input_indices, aspect_indices):
        texts = input_indices.cpu().numpy()
        aspects = aspect_indices.cpu().numpy()
        mask_matrix = np.ones((input_indices.size(0), self.opt.max_seq_len,
                               self.opt.bert_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(aspects))):
            aspect_len = np.count_nonzero(aspects[asp_i]) - 2
            aspect_begin = np.argwhere(texts[text_i] == aspects[asp_i][1])[0][0]
            lcf_begin = aspect_begin - self.opt.SRD if aspect_begin >= self.opt.SRD else 0
            for i in range(self.opt.max_seq_len):
                if i < lcf_begin or i > aspect_begin + aspect_len + self.opt.SRD - 1:
                    mask_matrix[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
        return torch.tensor(mask_matrix).to(self.opt.device)

    # create the weight matrix for generating local context features
    def get_weight_matrix(self, input_indices, aspect_indices):
        texts = input_indices.cpu().numpy()
        aspects = aspect_indices.cpu().numpy()
        weight_matrix = np.zeros((input_indices.size(0), self.opt.max_seq_len,
                                  self.opt.bert_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(aspects))):
            aspect_len = np.count_nonzero(aspects[asp_i]) - 2
            aspect_begin = np.argwhere(texts[text_i] == aspects[asp_i][1])[0][0]
            aspect_central_index = (aspect_begin * 2 + aspect_len) / 2
            weight_for_each_text = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(np.count_nonzero(texts[text_i])):
                weight_for_each_text[i] = 1 - (abs(i - aspect_central_index) + aspect_len / 2 - self.opt.SRD) /\
                np.count_nonzero(texts[text_i]) if abs(i - aspect_central_index) + aspect_len / 2 > self.opt.SRD else 1
                weight_matrix[text_i][i] = weight_for_each_text[i] * np.ones((self.opt.bert_dim))

        return torch.tensor(weight_matrix).to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        global_context_features, _ = self.bert4global(text_bert_indices, bert_segments_ids)
        local_context_features, _ = self.bert4local(text_local_indices)

        if self.opt.local_context_focus == 'cdm':
            mask_matrix = self.get_mask_matrix(text_local_indices, aspect_indices)
            cdm_features = torch.mul(local_context_features, mask_matrix)
            cdm_features = self.bert_SA(cdm_features)
            cat_features = torch.cat((cdm_features, global_context_features), dim=-1)
            cat_features = self.linear2(cat_features)

        elif self.opt.local_context_focus == 'cdw':
            weight_matrix = self.get_weight_matrix(text_local_indices, aspect_indices)
            cdw_features = torch.mul(local_context_features, weight_matrix)
            cdw_features = self.bert_SA(cdw_features)
            cat_features = torch.cat((cdw_features, global_context_features), dim=-1)
            cat_features = self.linear2(cat_features)

        elif self.opt.local_context_focus == 'lcf_fusion':
            mask_matrix = self.get_mask_matrix(text_local_indices, aspect_indices)
            cdm_features = torch.mul(local_context_features, mask_matrix)
            weight_matrix = self.get_weight_matrix(text_local_indices, aspect_indices)
            cdw_features = torch.mul(local_context_features, weight_matrix)
            cat_features = torch.cat((cdm_features, global_context_features, cdw_features), dim=-1)
            cat_features = self.linear3(cat_features)

        cat_features = self.dropout(cat_features)
        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)
        return dense_out
