# -*- coding: utf-8 -*-
# file: lcf_glove.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn

import numpy as np
from layers.point_wise_feed_forward import PositionwiseFeedForward
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

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


class LCF_GLOVE(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(LCF_GLOVE, self).__init__()
        self.config = BertConfig.from_json_file("config.json")
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.mha_global = SelfAttention(self.config, opt)
        self.mha_local = SelfAttention(self.config, opt)
        self.ffn_global = PositionwiseFeedForward(self.opt.hidden_dim, dropout=self.opt.dropout)
        self.ffn_local = PositionwiseFeedForward(self.opt.hidden_dim, dropout=self.opt.dropout)
        self.mha_local_SA = SelfAttention(self.config, opt)
        self.mha_global_SA = SelfAttention(self.config, opt)
        self.mha_SA_single = SelfAttention(self.config, opt)
        self.bert_pooler = BertPooler(self.config)

        self.dropout = nn.Dropout(opt.dropout)
        self.linear_triple = nn.Linear(opt.embed_dim * 3, opt.hidden_dim)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.hidden_dim)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        
    # create the mask matrix for generating local context features
    def get_mask_matrix(self, input_indices, aspect_indices):
        texts = input_indices.cpu().numpy()
        aspects = aspect_indices.cpu().numpy()
        mask_matrix = np.ones((input_indices.size(0), self.opt.max_seq_len,
                               self.opt.hidden_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(aspects))):
            aspect_len = np.count_nonzero(aspects[asp_i]) - 2
            aspect_begin = np.argwhere(texts[text_i] == aspects[asp_i][1])[0][0]
            lcf_begin = aspect_begin - self.opt.SRD if aspect_begin >= self.opt.SRD else 0
            for i in range(self.opt.max_seq_len):
                if i < lcf_begin or i > aspect_begin + aspect_len + self.opt.SRD - 1:
                    mask_matrix[text_i][i] = np.zeros((self.opt.hidden_dim), dtype=np.float)
        return torch.tensor(mask_matrix).to(self.opt.device)

    # create the weight matrix for generating local context features
    def get_weight_matrix(self, input_indices, aspect_indices):
        texts = input_indices.cpu().numpy()
        aspects = aspect_indices.cpu().numpy()
        weight_matrix = np.zeros((input_indices.size(0), self.opt.max_seq_len,
                                  self.opt.hidden_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(aspects))):
            aspect_len = np.count_nonzero(aspects[asp_i]) - 2
            aspect_begin = np.argwhere(texts[text_i] == aspects[asp_i][1])[0][0]
            aspect_central_index = (aspect_begin * 2 + aspect_len) / 2
            weight_for_each_text = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(np.count_nonzero(texts[text_i])):

                weight_for_each_text[i] = 1-(abs(i-aspect_central_index)+aspect_len/2-self.opt.SRD)/np.count_nonzero(texts[text_i]) \
                    if abs(i - aspect_central_index) + aspect_len / 2 > self.opt.SRD else 1
                weight_matrix[text_i][i] = weight_for_each_text[i] * np.ones((self.opt.hidden_dim))

        return torch.tensor(weight_matrix).to(self.opt.device)

    def forward(self, inputs):

        text_global_indices = inputs[0]
        text_local_indices = inputs[1]
        aspect_indices = inputs[2]

        # embedding layer
        global_context_features = self.embed(text_global_indices)
        local_context_features = self.embed(text_local_indices)

        # PFE layer
        global_context_features = self.mha_global(global_context_features)
        local_context_features = self.mha_local(local_context_features)
        global_context_features = self.ffn_global(global_context_features)
        local_context_features = self.ffn_local(local_context_features)

        # dropout
        global_context_features = self.dropout(global_context_features).to(self.opt.device)
        local_context_features = self.dropout(local_context_features).to(self.opt.device)

        # LCF layer
        if self.opt.local_context_focus == 'cdm':
            mask_matrix = self.get_mask_matrix(text_local_indices, aspect_indices)
            local_context_features = torch.mul(local_context_features, mask_matrix)
        elif self.opt.local_context_focus == 'cdw':
            weight_matrix = self.get_weight_matrix(text_local_indices, aspect_indices)
            local_context_features = torch.mul(local_context_features, weight_matrix)
        elif self.opt.local_context_focus == 'lcf_fusion':
            mask_matrix = self.get_mask_matrix(text_local_indices, aspect_indices)
            cdm_features = torch.mul(global_context_features, mask_matrix)
            cdw_features = self.get_weight_matrix(text_local_indices, aspect_indices)
            lcf_fusion_features = torch.mul(global_context_features, cdw_features)
            out_cat = torch.cat((cdm_features, global_context_features, lcf_fusion_features), dim=-1)
            local_context_features = self.linear_triple(out_cat)

        lcf_features = self.mha_local_SA(local_context_features)
        global_context_features = self.mha_global_SA(global_context_features)
        # FIL layer
        cat_out = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_out = self.linear(cat_out)
        cat_out = self.mha_SA_single(cat_out)

        # output layer
        pooled_out = self.bert_pooler(cat_out)
        dense_out = self.dense(pooled_out)
        return dense_out
