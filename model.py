"""
The code is copied and adapted from https://github.com/Ac-Zyx/RoCORE
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def finetune(model, unfreeze_layers):
    params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if params_name_mapping[ele] in name:
                param.requires_grad = True
                break
    return model

class Margin():
    def __init__(self, args, dict):
        from taxo import TaxStruct
        import codecs
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            tax_lines = f.readlines()
        self.tax_pairs = [line.strip().split(" ") for line in tax_lines]
        self.tax_graph = TaxStruct(self.tax_pairs)
        self.nodes = list(self.tax_graph.nodes.keys())
        self.dict = dict

    def get_margin(self, list_a, list_b):
        margin = []
        for k, i in enumerate(list_a):
            node_a = self.dict[i.item()]
            path_a = self.tax_graph.node2path[node_a]
            node_b = self.dict[list_b[k].item()]
            path_b = self.tax_graph.node2path[node_b]
            com = len(set(path_a).intersection(set(path_b)))
            m = max( min(( abs(len(path_a) - com) + abs(len(path_b) - com) ) / com, 2), 0.5 )
            margin.append(m)
        return margin

class HALTON(nn.Module): # Pooling as relation description representation
    def __init__(self, args, config, pretrained_model, unfreeze_layers = []):
        super().__init__()
        self.max_len = args.max_len
        self.num_class = args.num_class
        self.new_class = args.new_class
        self.hidden_dim = args.hidden_dim
        self.kmeans_dim = args.kmeans_dim
        self.initial_dim = config.hidden_size
        assert config.output_hidden_states == True
        self.unfreeze_layers = unfreeze_layers
        self.pretrained_model = finetune(pretrained_model, self.unfreeze_layers) # fix bert weights
        self.layer = args.layer
        self.similarity_encoder = nn.Sequential(
                nn.Linear(self.initial_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.kmeans_dim)
        )
        self.similarity_decoder = nn.Sequential(
                nn.Linear(self.kmeans_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.initial_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(self.initial_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.kmeans_dim)
        )
        self.margin_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.initial_dim, self.kmeans_dim)
        )
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.ct_loss_u = CenterLoss_unlabel(dim_hidden = self.kmeans_dim, num_classes = self.new_class)
        self.ct_loss_l = CenterLoss_label(dim_hidden = self.kmeans_dim, num_classes = self.num_class)
        self.ce_loss = nn.CrossEntropyLoss()

        self.labeled_head = nn.Linear(self.initial_dim, self.num_class)
        self.unlabeled_head = nn.Linear(self.initial_dim, self.new_class)

        self.bert_params = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad is True:
                self.bert_params.append(param)
                
    @torch.no_grad()
    def normalize_head(self):
        w = self.labeled_head.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.labeled_head.weight.copy_(w)

        w = self.unlabeled_head.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.unlabeled_head.weight.copy_(w)

    def get_pretrained_feature(self, input_ids, input_mask, valid_mask, pos_span, mask_span, layer = 12):
        output = self.pretrained_model(input_ids, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        encoder_layers = output.hidden_states[layer] # (batch_size, seq_len, bert_embedding)
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
        # print(pretrained_feat.shape)
        return pretrained_feat, mask_feat
        
    def forward(self, data, msg = 'feat', using_mask = False):
        if msg == 'con':
            input_ids, input_mask,valid_mask, label, pos_span, mask_span = data
            input_ids, input_mask, valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
            pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask, valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                logits = self.head(mask_feat) 
            else:
                logits = self.head(pretrained_feat) 
            return logits 

        elif msg == "margin":
            input_ids, input_mask,valid_mask, label, pos_span, mask_span = data
            input_ids, input_mask, valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
            pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask, valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                logits = self.margin_head(mask_feat) 
            else:
                logits = self.margin_head(pretrained_feat) 
            return logits 

        elif msg == 'similarity':
            with torch.no_grad():
                input_ids, input_mask,valid_mask, label, pos_span, mask_span = data
                input_ids, input_mask,valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
                pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask,valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                sia_rep = self.similarity_encoder(mask_feat) # (batch_size, keamns_dim)
            else:
                sia_rep = self.similarity_encoder(pretrained_feat) # (batch_size, keamns_dim)
            return sia_rep # (batch_size, keamns_dim)

        elif msg == 'reconstruct':
            with torch.no_grad():
                input_ids, input_mask,valid_mask, label, pos_span, mask_span = data
                input_ids, input_mask,valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
                pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask, valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                sia_rep = self.similarity_encoder(mask_feat) # (batch_size, kmeans_dim)
                rec_rep = self.similarity_decoder(sia_rep) # (batch_size, 2 * bert_embedding)
                rec_loss = (rec_rep - mask_feat).pow(2).mean(-1)
            else:
                sia_rep = self.similarity_encoder(pretrained_feat) # (batch_size, kmeans_dim)
                rec_rep = self.similarity_decoder(sia_rep) # (batch_size, 2 * bert_embedding)
                rec_loss = (rec_rep - pretrained_feat).pow(2).mean(-1)
            return sia_rep, rec_loss


        elif msg == 'labeled':
            input_ids, input_mask,valid_mask, label, pos_span, mask_span = data
            input_ids, input_mask, valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
            pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask, valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                logits = self.labeled_head(mask_feat) 
            else:
                logits = self.labeled_head(pretrained_feat)
            return logits # (batch_size, num_class + new_class)

        elif msg == 'unlabeled':
            input_ids, input_mask, valid_mask, label, pos_span, mask_span = data
            input_ids, input_mask,valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
            pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask, valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                logits = self.unlabeled_head(mask_feat)
            else:
                logits = self.unlabeled_head(pretrained_feat)
            return logits # (batch_size, new_class)

        elif msg == 'feat':
            input_ids, input_mask, valid_mask, label, pos_span, mask_span = data
            input_ids, input_mask,valid_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, valid_mask, label)))
            pretrained_feat, mask_feat = self.get_pretrained_feature(input_ids, input_mask,valid_mask, pos_span, mask_span) # (batch_size, 2 * bert_embedding)
            if using_mask:
                return mask_feat
            else:
                return pretrained_feat

        else:
            raise NotImplementedError('not implemented!')

class CenterLoss_label(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 0.1

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances) # (batch_size) => scalar
        return loss


class CenterLoss_unlabel(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 1.

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        q = 1.0/(1.0+distance_centers/self.alpha) # (batch_size, num_class)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        prob = q.gather(1, y.unsqueeze(1)).squeeze() # (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances*prob) # (batch_size) => scalar
        return loss



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf. 
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


