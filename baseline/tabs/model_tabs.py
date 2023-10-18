from typing import List, Dict, Tuple, Optional 
from copy import deepcopy 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import os 
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl



from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm 

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from functools import wraps
from time import time

from typing import List, Optional, Tuple, Dict
from collections import OrderedDict 
from math import ceil
import torch
from torch import nn   
import torch.nn.functional as F 
from scipy.optimize import linear_sum_assignment
from copy import deepcopy 

def onedim_gather(src: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
    '''
    src: (batch, M, L)
    index: (batch, M) or (batch, L) or (batch, 1)

    A version of the torch.gather function where the index is only along 1 dim.
    '''
    for i in range(len(src.shape)):
        if i!=0 and i!=dim:
            # index is missing this dimension
            index = index.unsqueeze(dim=i)
    target_index_size = deepcopy(list(src.shape))
    target_index_size[dim] = 1 
    index = index.expand(target_index_size)
    output = torch.gather(src, dim, index)
    return output 


class Prototypes(nn.Module):
    def __init__(self, feat_dim, num_prototypes, norm:bool=False):
        super().__init__()

        self.prototypes = nn.Linear(feat_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def initialize_prototypes(self, centers):
        self.prototypes.weight.copy_(centers)
        return 
    

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    
    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    '''
    Simple n layer MLP with ReLU activation and batch norm.
    The order is Linear, Norm, ReLU 
    '''
    def __init__(self, feat_dim: int, hidden_dim: int,  latent_dim:int, 
        norm:bool=False, norm_type:str='batch', layers_n: int = 1, dropout_p: float =0.1 ):
        '''
        :param norm_type: one of layer, batch 
        '''
        super().__init__() 
        self.feat_dim= feat_dim 
        self._hidden_dim= hidden_dim
        self.latent_dim = latent_dim 
        self.input2hidden = nn.Linear(feat_dim, hidden_dim) 
        self.dropout = nn.Dropout(p=dropout_p)       
        layers = [self.dropout, ]
        for i in range(layers_n):
            if i==0:
                layers.append(nn.Linear(feat_dim, hidden_dim))
                out_dim = hidden_dim 
            elif i==1:
                layers.append(nn.Linear(hidden_dim, latent_dim))
                out_dim = latent_dim 
            else:
                layers.append(nn.Linear(latent_dim, latent_dim))
                out_dim = latent_dim 
            if norm:
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                else:
                    layers.append(nn.LayerNorm(out_dim))
            if i < layers_n -1: # last layer has no relu 
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        '''
        :param input: torch.FloatTensor (batch, ..., feat_dim) 

        :return output: torch.FloatTensor (batch, ..., hidden_dim)
        '''
        output = self.net(input.reshape(-1, self.feat_dim))
       
        original_shape = input.shape 
        new_shape = tuple(list(input.shape[:-1]) + [self.latent_dim])

        output = output.reshape(new_shape) 
        return output 

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, queue_len:int=1024, classes_n: int=10, delta=0.0):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.delta = delta

        self.classes_n = classes_n
        self.queue_len = queue_len
        self.register_buffer(name='logit_queue', tensor=torch.zeros((queue_len, classes_n)), persistent=False)
        self.cur_len = 0 

    def add_to_queue(self, logits: torch.FloatTensor)-> None:
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        classes_n = logits.size(1)
        assert (classes_n == self.classes_n)

        new_queue = torch.concat([logits, self.logit_queue], dim=0)
        self.logit_queue = new_queue[:self.queue_len, :]

        self.cur_len += batch_size
        
        self.cur_len = min(self.cur_len, self.queue_len)

        return 

    def queue_full(self)-> bool:
        
        return self.cur_len == self.queue_len



    @torch.no_grad()
    def forward(self, logits: torch.FloatTensor):
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        all_logits = self.logit_queue 
        
        initial_Q = torch.softmax(all_logits/self.epsilon, dim=1) 
        # Q = torch.exp(logits / self.epsilon).t() # (K, N)
        Q = initial_Q.clone().t() 
        N = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        assert (torch.any(torch.isinf(sum_Q)) == False), "sum_Q is too large"
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows += self.delta # for numerical stability 
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols 
            Q /= N

        Q *= N  # the colomns must sum to 1 so that Q is an assignment

        batch_assignments = Q.t()[:batch_size, :]
        return batch_assignments, sum_of_rows.squeeze(), sum_of_cols.squeeze()  

class CommonSpaceCache(nn.Module):
    '''
    A cache for saving common space embeddings and using it to compute contrastive loss.
    '''
    def __init__(self, feature_size:int, known_cache_size: int, unknown_cache_size: int, metric_type: str='cosine', sim_thres: float=0.8) -> None:
        super().__init__()
        self.feature_size = feature_size 
        self.known_cache_size = known_cache_size
        self.unknown_cache_size = unknown_cache_size
        self.known_len=0
        self.unknown_len =0 

        self.metric_type =metric_type 
        self.metric = nn.CosineSimilarity(dim=2, eps=1e-8)
        self.sim_thres=sim_thres 
       
        self.temp = 0.1 # temperature for softmax 

        self.register_buffer("known_cache", torch.zeros((known_cache_size, feature_size), dtype=torch.float), persistent=False)
        self.register_buffer("unknown_cache", torch.zeros((unknown_cache_size, feature_size), dtype=torch.float), persistent=False)

        self.register_buffer("known_labels", torch.zeros((known_cache_size,), dtype=torch.long), persistent=False)
        self.register_buffer("unknown_labels", torch.zeros((unknown_cache_size, ), dtype=torch.long), persistent=False)


    def cache_full(self)-> bool:
        if (self.known_len == self.known_cache_size) and (self.unknown_len == self.unknown_cache_size):
            return True 
        else:
            return False 

    @torch.no_grad() 
    def update_batch(self, embeddings: torch.FloatTensor, known_mask: torch.BoolTensor, labels: Optional[torch.LongTensor]=None) -> None:
        '''
        Add embeddings to cache.
        :param embeddings: (batch, feature_size)
        '''
        embeddings_detached = embeddings.detach() 

        known_embeddings = embeddings_detached[known_mask,:]
        known_size = known_embeddings.size(0)
        new_known_cache = torch.concat([known_embeddings, self.known_cache], dim=0)
        self.known_cache = new_known_cache[:self.known_cache_size]
        self.known_len = min(self.known_len + known_size, self.known_cache_size)
        if labels!=None: 
            known_labels = labels[known_mask] 
            self.known_labels = torch.concat([known_labels, self.known_labels], dim=0)[:self.known_cache_size]
            unknown_labels = labels[~known_mask]
            self.unknown_labels = torch.concat([unknown_labels, self.unknown_labels], dim=0)[:self.unknown_cache_size]


        unknown_embeddings = embeddings_detached[~known_mask,: ]
        unknown_size = unknown_embeddings.size(0)
        new_unknown_cache = torch.concat([unknown_embeddings, self.unknown_cache], dim=0)
        self.unknown_cache = new_unknown_cache[:self.unknown_cache_size]
        self.unknown_len = min(self.unknown_len + unknown_size, self.unknown_cache_size)
        return 

    @torch.no_grad()
    def get_positive_example(self, embedding: torch.FloatTensor, known: bool =False) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        '''
        :param embeddings (N, feature_dim)

        :returns (N, feature_dim)
        '''
        embedding_detached = embedding.detach() 
        if known: 
            cache = self.known_cache
            label_cache = self.known_labels
        else:
            cache = self.unknown_cache 
            label_cache = self.unknown_labels

        if self.metric_type == 'cosine':       
            similarity = self.metric(embedding_detached.unsqueeze(dim=1), cache.unsqueeze(dim=0)) # N, cache_size
        else:
            similarity = torch.einsum("ik,jk->ij", embedding_detached, cache) 
        
        max_sim, max_idx = torch.max(similarity, dim=1) #(N, )
        min_thres = self.sim_thres 
        valid_pos_mask = (max_sim > min_thres) #(N, )
        pos_embeddings = cache[max_idx, :] # (N, feature_dim)
        pos_labels = label_cache[max_idx] # (N, )

        return pos_embeddings, valid_pos_mask, pos_labels 
    
    @torch.no_grad() 
    def get_negative_example_for_unknown(self, embedding: torch.FloatTensor, k: int=3) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Take half of the negative examples from the unknown cache and half from the known cache.
        :param embeddings (N, feature_dim)
        '''
        embedding_detached= embedding.detach() 
        N = embedding_detached.size(0)
        if self.metric_type == 'cosine':
            unknown_similarity = self.metric(embedding_detached.unsqueeze(dim=1), self.unknown_cache.unsqueeze(dim=0)) # N, cache_size
        else:
            unknown_similarity = torch.einsum('ik,jk->ij', embedding_detached, self.unknown_cache)
        
        sorted_unk_idx = torch.argsort(unknown_similarity, dim=1) # N, cache_size
        unk_n = ceil(sorted_unk_idx.size(1) /2)
        candidate_neg_unk_idx = sorted_unk_idx[:, :unk_n] # N, cache_size/2
        # this is used for generating indexes 
        neg_unk_list = []
        for i in range(N):
            random_idx = torch.randperm(n=unk_n, dtype=torch.long, device=embedding.device)[:k]
            chosen_neg_unk_idx = candidate_neg_unk_idx[i, :][random_idx]
            chosen_neg_unk = self.unknown_cache[chosen_neg_unk_idx, :] # K, feature_size 
            neg_unk_list.append(chosen_neg_unk)
        
        if self.metric_type == 'cosine':
            known_similarity = self.metric(embedding_detached.unsqueeze(dim=1), self.known_cache.unsqueeze(dim=0)) # (N, cache_size)
        else:
            known_similarity = torch.einsum("ik,jk->ij", embedding_detached, self.known_cache)
        
        sorted_known_idx = torch.argsort(known_similarity, dim=1, descending=True) # choose hard examples (N, cache_size)
        neg_known_list = []
        chosen_neg_known_idx = sorted_known_idx[:, :k]
        for i in range(N):
            chosen_neg_known = self.known_cache[chosen_neg_known_idx[i], :]
            neg_known_list.append(chosen_neg_known)

        neg_unk = torch.stack(neg_unk_list, dim=0)
        neg_known = torch.stack(neg_known_list, dim=0) # (N, K, feature_size)
            
        return neg_unk, neg_known

    def get_contrastive_candidates(self, embeddings: torch.FloatTensor, neg_n: int=6, labels: Optional[torch.LongTensor]=None):
        N = embeddings.size(0)
        if labels!=None: assert (labels.size(0) == N)

        pos_embeddings, valid_pos_mask, pos_labels  = self.get_positive_example(embeddings, known=False) # (N, hidden_dim)
        assert (pos_embeddings.shape == embeddings.shape )
        # report positive sample accuracy 
        pos_acc = self.compute_accuracy(labels[valid_pos_mask], pos_labels[valid_pos_mask])

        neg_unk_embeddings, neg_known_embeddings  = self.get_negative_example_for_unknown(embeddings, k=ceil(neg_n/2)) # (N, K, hidden_dim)
        candidates = torch.concat([pos_embeddings.unsqueeze(dim=1), neg_unk_embeddings, neg_known_embeddings], dim=1) # (N, 2K+1, hidden_dim)
        # scores = torch.einsum('ik,ijk->ij', embeddings, candidates) # (N, 2K+1 )
        # targets = torch.zeros((N,), dtype=torch.long, device=scores.device)
        # loss = F.cross_entropy(scores/self.temp, targets)
        return candidates, valid_pos_mask, pos_acc 

    def compute_accuracy(self, labels, other_labels):
        # consider moving average 
        assert (labels.shape == other_labels.shape)
        acc = torch.sum(labels == other_labels)*1.0 / labels.size(0)
        return acc 


class ClassifierHead(nn.Module):
    def __init__(self, args, feature_size: int, 
            n_classes: int, layers_n: int = 1, 
            n_heads: int =1, dropout_p: float =0.2, hidden_size: Optional[int]=None) -> None:
        super().__init__()
        self.args = args 

        self.feature_size = feature_size 
        self.n_classes = n_classes 
        self.n_heads = n_heads
        if hidden_size: 
            self.hidden_size = hidden_size 
        else:
            self.hidden_size = feature_size 

        if layers_n == 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('dropout',nn.Dropout(p=dropout_p)),
                ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
                ))
        elif layers_n > 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('mlp', MLP(feat_dim=self.feature_size, hidden_dim=self.hidden_size, latent_dim=self.hidden_size, norm=True,layers_n=layers_n-1)),
                ('dropout',nn.Dropout(p=dropout_p)),
                ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
                ))

    def initialize_centroid(self, centers):
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.initialize_prototypes(centers)
        
        return 

    def update_centroid(self):
        '''
        The centroids are essentially just the vectors in the final Linear layer. Here we normalize them. they are trained along with the model.
        '''
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.normalize_prototypes()

        return 
    
    def freeze_centroid(self):
        '''
        From Swav paper, freeze the prototypes to help with initial optimization.
        '''
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.freeze_prototypes() 

        return 

    def unfreeze_centroid(self):
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.unfreeze_prototypes() 

        return 


    def forward(self, inputs: torch.FloatTensor):
        '''
        :params inputs: (batch, feat_dim)

        :returns logits: (batch, n_classes)
        '''
        outputs = self.classifier(inputs)
        return outputs 

class MultiviewModel(nn.Module):
    def __init__(self, args, model_config, pretrained_model, unfreeze_layers: List=[]) -> None:
        super().__init__()
        self.args = args 
        
        self.layer = args.layer 
        if args.freeze_pretrain:
            self.pretrained_model = self.finetune(pretrained_model, unfreeze_layers)  
        else:
            self.pretrained_model = pretrained_model 

        self.views = nn.ModuleList() 
        if args.feature == 'all': feature_types = ['token','mask']
        elif args.feature == 'mask': feature_types = ['mask', 'mask']
        elif args.feature == 'token': feature_types = ['token', 'token']
        else: 
            raise NotImplementedError

        known_head_types = args.num_class 
        
        for view_idx, ft in enumerate(feature_types):
            if ft == 'mask':
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(model_config.hidden_size, args.hidden_dim, args.kmeans_dim,
                    norm=True, norm_type='batch', layers_n=2, dropout_p =0.1),
                        'known_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    known_head_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    args.new_class, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim)
                    }     
                )
            else:

                input_size = model_config.hidden_size # trigger 
                
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(input_size, args.hidden_dim, args.kmeans_dim,
                    norm=True, norm_type='batch', layers_n=2, dropout_p=0.1),
                        'known_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    known_head_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    args.new_class, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim)
                    }
                )
            self.views.append(view_model)

        # this commonspace means that known classes and unknown classes are projected into the same space 
        self.commonspace_cache = nn.ModuleList([
            CommonSpaceCache(feature_size=args.kmeans_dim, known_cache_size=512, unknown_cache_size=256, sim_thres=0.8),
            CommonSpaceCache(feature_size=args.kmeans_dim, known_cache_size=512, unknown_cache_size=256, sim_thres=0.8)
        ])

        self.device = torch.device("cuda" if args.cuda else "cpu")
        return 

    @staticmethod
    def finetune(model, unfreeze_layers):
        params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if params_name_mapping[ele] in name:
                    param.requires_grad = True
                    break
        return model

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['token_ids'].to(self.device), 'attention_mask': batch['attn_mask'].to(self.device)}
        return inputs

    def predict_name(self, batch:Dict, topk: int=10):
        input_ids, input_mask, valid_mask, label, pos_span, mask_span = batch
        input_ids, input_mask, valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
        outputs = self.pretrained_model(input_ids, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        vocab_logits = outputs[0] #(batch_size, seq, vocab_size)
        mask_bpe_idx = mask_span.cuda()
        mask_logits = onedim_gather(vocab_logits, dim=1, index=mask_bpe_idx.unsqueeze(1)).squeeze(1) # (batch_size, vocab_size)
        # print(vocab_logits.shape)
        predicted_token_ids = mask_logits.argsort(dim=-1, descending=True)[:, :topk]
        return predicted_token_ids
        
    def _compute_features(self, batch)-> torch.FloatTensor:
        input_ids, input_mask, valid_mask, label, pos_span, mask_span = batch
        input_ids, input_mask, valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
        output = self.pretrained_model(input_ids, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        encoder_layers = output.hidden_states[self.layer] # (batch_size, seq_len, bert_embedding)
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
        pretrained_feat = torch.stack([torch.max(valid_output[i, pos_span[i][0]:pos_span[i][1]+1, :], dim = 0)[0] for i in range(batch_size)], dim = 0)

        return pretrained_feat, mask_feat
    
    def _compute_prediction_logits(self, batch, is_l = True,
             view_idx: int =0)-> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pretrained_feat, mask_feat = self._compute_features(batch)
        
        if is_l == True:
            p_feat = self.views[0]['common_space_proj'](pretrained_feat)
            pretrained_logit = self.views[0]['known_type_classifier'](p_feat)
            m_feat = self.views[1]['common_space_proj'](mask_feat)
            mask_logit = self.views[1]['known_type_classifier'](m_feat)
        else:
            p_feat = self.views[0]['common_space_proj'](pretrained_feat)
            pretrained_logit = self.views[0]['unknown_type_classifier'](p_feat)
            m_feat = self.views[1]['common_space_proj'](mask_feat)
            mask_logit = self.views[1]['unknown_type_classifier'](m_feat)
  

        return pretrained_logit, mask_logit

    def _on_train_batch_start(self):
         # normalize all centroids 
        for view_model in self.views: 
            view_model['known_type_classifier'].update_centroid() 
            view_model['unknown_type_classifier'].update_centroid() 

        return 
    


class TypeDiscoveryModel(nn.Module):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.config = args  
        
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.bert_model, output_hidden_states = True)

        # AutoConfig.from_pretrained(args.bert_model, output_hidden_states = True)
        if args.name:
            pretrained_model = AutoModelForMaskedLM.from_pretrained(args.bert_model, config=self.model_config)
        else:
            pretrained_model = AutoModel.from_pretrained(args.bert_model, config = self.model_config)
        embeddings = pretrained_model.resize_token_embeddings(len(self.tokenizer)) # when adding new tokens, the tokenizer.vocab_size is not changed! 

        self.mv_model = MultiviewModel(args, self.model_config, pretrained_model, unfreeze_layers= [args.layer])
        # regularization
        self.sk = SinkhornKnopp(num_iters=3, epsilon=self.config.sk_epsilon, classes_n= args.new_class, queue_len=1024, delta=1e-10)
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.current_epoch = 0



    def _smooth_targets(self, targets:torch.FloatTensor, all_types: int):
        if self.config.label_smoothing_alpha > 0.0:
            if self.config.label_smoothing_ramp > 0:
                alpha =( 1- self.current_epoch*1.0/self.config.label_smoothing_ramp) * self.config.label_smoothing_alpha
            else:
                alpha = self.config.label_smoothing_alpha

            alpha = max(alpha, 0.0)
            targets = (1-alpha) * targets + alpha * torch.full_like(targets, fill_value=1.0/all_types, dtype=torch.float, device=self.device)
        
        return targets 

    def _compute_targets(self, batch_size:int, labels: torch.LongTensor, 
            predicted_logits:torch.FloatTensor, predicted_logits_other: torch.FloatTensor, hard: bool=False):

        targets = torch.zeros((batch_size, self.config.num_class), dtype=torch.float, device=self.device) # soft targets 
        all_types= self.config.num_class
        assert (labels.max() < all_types) 
        known_labels = F.one_hot(labels, num_classes=all_types).float() # (batch, all_types)

        targets = known_labels
        targets_other = targets 
    
        targets = self._smooth_targets(targets, all_types)
        targets_other = self._smooth_targets(targets_other, all_types)

        return targets, targets_other 

    def _compute_batch_pairwise_loss(self,
        predicted_logits: torch.FloatTensor, 
        labels: Optional[torch.LongTensor]=None,
        loss_fn: str='kl', sigmoid:float =2.0):
    
        if self.config.rev_ratio > 0: 
            num_class = self.config.num_class *2 
        else:
            num_class = self.config.num_class

        # predicted_logits = predicted_logits[:, num_class:] 

        def compute_kld(p_logit, q_logit):
            p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
            q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
            return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)
    

        batch_size = labels.size(0)
        label_mask = (labels != -1)
        
        pairwise_label = (labels[label_mask].unsqueeze(0) == labels[label_mask].unsqueeze(1)).float()
        predicted_logits = predicted_logits[label_mask]


        expanded_logits = predicted_logits.expand(batch_size, -1, -1)
        expanded_logits2 = expanded_logits.transpose(0, 1)
        kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
        kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)
        pair_loss = torch.mean(pairwise_label * (kl1 + kl2) + (1 - pairwise_label) * (torch.relu(sigmoid - kl1) + torch.relu(sigmoid - kl2)))
        return pair_loss 

    def _compute_consistency_loss(self, predicted_logits: torch.FloatTensor, 
            predicted_logits_other: torch.FloatTensor, 
            loss_fc='kl'):
        assert (predicted_logits.shape == predicted_logits_other.shape)

        def compute_kld(p_logit, q_logit):
            p = F.softmax(p_logit, dim = -1) # (B, n_class) 
            q = F.softmax(q_logit, dim = -1) # (B, n_class)
            return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B,)
        
        kl1 = compute_kld(predicted_logits.detach(), predicted_logits_other)
        kl2 = compute_kld(predicted_logits_other.detach(), predicted_logits)

        consistency_loss = torch.mean( kl1+kl2) * 0.5
    
        return consistency_loss 


    def forward(self, batch, epoch ,is_l = True):
        labels = batch[3]
        pseudo1 = batch[-2]
        pseudo2 = batch[-1]
        batch_size = labels.size(0)
        if is_l:
            pretrained_feat, mask_feat = self.mv_model._compute_prediction_logits(batch[:-1], is_l)
        else:
            pretrained_feat, mask_feat = self.mv_model._compute_prediction_logits(batch[:-3], is_l)
        
        if is_l:
            targets, _ = self._compute_targets(batch_size, labels, pretrained_feat, mask_feat, hard=False)
            loss = F.cross_entropy(pretrained_feat, target=targets) + F.cross_entropy(mask_feat, target=targets)
        else:
            loss = self._compute_batch_pairwise_loss(pretrained_feat,labels=pseudo2, loss_fn='kl',sigmoid=self.config.sigmoid) \
                + self._compute_batch_pairwise_loss(mask_feat, labels=pseudo1, loss_fn='kl', sigmoid=self.config.sigmoid)

        consistency_loss = self._compute_consistency_loss(pretrained_feat, mask_feat, loss_fc='kl')
        loss += consistency_loss
        return loss


    def encoder(self, batch, is_l=False):
        pretrained_feat, mask_feat = self.mv_model._compute_prediction_logits(batch[:-3], is_l)
        
            
        if self.config.psuedo_label == 'combine':
            target_logits = pretrained_feat + mask_feat
        elif self.config.psuedo_label == 'self':
            target_logits = pretrained_feat
        else:
            target_logits = mask_feat

        return target_logits

    def get_feat(self, data, is_l = False):
        if is_l:
            pretrained_feat, mask_feat = self.mv_model._compute_features(data[:-1])
        else:
            pretrained_feat, mask_feat = self.mv_model._compute_features(data[:-3])
        return pretrained_feat, mask_feat

    def get_cluster_feat(self, data, is_l = False):
        if is_l:
            pretrained_feat, mask_feat = self.mv_model._compute_features(data[:-1])
        else:
            pretrained_feat, mask_feat = self.mv_model._compute_features(data[:-3])
        p_feat =  self.mv_model.views[0]['common_space_proj'](pretrained_feat)
        m_feat = self.mv_model.views[1]["common_space_proj"](mask_feat)
        return p_feat, m_feat

    def get_name(self,data):
        predicted_token_ids = self.mv_model.predict_name(data[:-3])
        # print(predicted_token_ids)
        predict_name = self.tokenizer.batch_decode(predicted_token_ids)
        return predict_name

