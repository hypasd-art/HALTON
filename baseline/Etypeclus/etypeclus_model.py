import os 
import json 
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm 

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup



def cosine_dist(x_bar: torch.FloatTensor, x: torch.FloatTensor, weight=None):
    if weight is None:
        weight = torch.ones(x.size(0), device=x.device)
    cos_sim = (x_bar * x).sum(-1)
    cos_dist = 1 - cos_sim
    cos_dist = (cos_dist * weight).sum() / weight.sum()
    return cos_dist

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        '''
        :param input_dim1: size of first input 
        :param hidden_dims: list of dimension sizes 
        '''
        super().__init__()

        print("hidden_dims:", hidden_dims)
        self.encoder_layers = []
        dims = [input_dim, ] + hidden_dims
        for i in range(len(dims) - 1):
            if i == 0:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            elif i != 0 and i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim,]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        x_bar = F.normalize(x_bar, dim=-1)
        return x_bar, z 

class ETypeClusModel(nn.Module):
    '''
    This is a wrapper of the TopicCluster class.
    '''
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.config = args 
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.bert_model, output_hidden_states = True)
        self.pretrained_model = AutoModel.from_pretrained(args.bert_model, config = self.model_config)
        self.temperature = args.temperature
        self.distribution = args.distribution

        input_dim = self.model_config.hidden_size 
        hidden_dims = eval(args.hidden_dims)
        self.topic_emb = nn.Parameter(torch.Tensor(args.new_class, hidden_dims[-1]))
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.q_dict = {} # uid -> target distribution 

        self.model = AutoEncoder(input_dim, hidden_dims)
        
        torch.nn.init.xavier_normal_(self.topic_emb.data)

        self.freeze_model() 

    def freeze_model(self):
        self.pretrained_model.requires_grad_(False)
        return 


    def cluster_assign(self, z):
        '''
        :param z: (batch, hidden_dim)
        :returns p: (batch, n_clusters) 
        '''
        if self.distribution == 'student':
            p = 1.0 / (1.0 + torch.sum(
                torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2) / self.alpha)
            p = p.pow((self.alpha + 1.0) / 2.0)
            p = (p.t() / torch.sum(p, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            p = F.softmax(sim, dim=-1)
        return p
    
    def forward_feat(self, x):
        x_bar, z = self.model(x)
        p = self.cluster_assign(z)
        return x_bar, z, p

    def target_distribution(self, z, 
         freq, method='all', top_num=0):
        '''
        :param x: (batch, hidden_dim)
        :param freq: (batch)
        ''' 
        p = self.cluster_assign(z).detach()
        if method == 'all':
            q = p**2 / (p * freq.unsqueeze(-1)).sum(dim=0)
            q = (q.t() / q.sum(dim=1)).t()
        elif method == 'top':
            assert top_num > 0
            q = p.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                q[topic_idx] = 0
                q[topic_idx, i] = 1
        return p, q

    def encoder(self, data):
        input_ids, input_mask, valid_mask, labels, pos_span, mask_span = data
        input_ids, input_mask, valid_mask, labels = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, labels)))
        output = self.pretrained_model(input_ids, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        encoder_layers = output.hidden_states[self.config.layer] # (batch_size, seq_len, bert_embedding)
        batch_size = encoder_layers.size(0)
        max_len = encoder_layers.size(1)
        feat_dim = encoder_layers.size(2)
        mask_feat = torch.stack([encoder_layers[i, mask_span[i]:mask_span[i]+1, :] for i in range(batch_size)], dim = 0).squeeze(1)
        # print(mask_feat.shape)
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            pos = 0
            for j in range(max_len):
                if valid_mask[i][j].item() == 1:
                    valid_output[i][pos] = encoder_layers[i][j]
                    pos += 1
        feat = torch.stack([torch.max(valid_output[i, pos_span[i][0]:pos_span[i][1]+1, :], dim = 0)[0] for i in range(batch_size)], dim = 0)
        
        x = F.normalize(feat) 

        x_bar, z , p = self.forward_feat(x)
        # z: the vector in hidden space
        # p: the possibility of the distribution
        return z, p

    def forward(self, data, data_idx, is_pretrain=False):
        input_ids, input_mask, valid_mask, labels, pos_span, mask_span = data
        input_ids, input_mask, valid_mask, labels = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, labels)))
        output = self.pretrained_model(input_ids, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        encoder_layers = output.hidden_states[self.config.layer] # (batch_size, seq_len, bert_embedding)
        batch_size = encoder_layers.size(0)
        max_len = encoder_layers.size(1)
        feat_dim = encoder_layers.size(2)
        mask_feat = torch.stack([encoder_layers[i, mask_span[i]:mask_span[i]+1, :] for i in range(batch_size)], dim = 0).squeeze(1)

        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            pos = 0
            for j in range(max_len):
                if valid_mask[i][j].item() == 1:
                    valid_output[i][pos] = encoder_layers[i][j]
                    pos += 1
        feat = torch.stack([torch.max(valid_output[i, pos_span[i][0]:pos_span[i][1]+1, :], dim = 0)[0] for i in range(batch_size)], dim = 0)
        
        x = F.normalize(feat) 

        x_bar, z , p = self.forward_feat(x)
        reconstr_loss = cosine_dist(x_bar, x) 
        
        if is_pretrain: 
            loss = reconstr_loss
            return loss
        
        q_batch = torch.zeros((batch_size, self.config.new_class), dtype=torch.float, device=self.device)
        for i in range(batch_size):
            uid = data_idx[i].item()
            q_batch[i, : ] = self.q_dict[uid] 
        kl_loss = F.kl_div(p.log(), q_batch, reduction='none').sum()
        loss = self.config.gamma * kl_loss + reconstr_loss
      
        return loss 

