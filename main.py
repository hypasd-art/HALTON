from dataset import Labeled_Dataset as BertDataset
from dataset import unLabeled_Dataset as PBertDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import *
from utils import seed_everything, data_split2, L2Reg, compute_kld, _worker_init_fn_,build_vocab
import random
import os
from evaluation import ClusterEvaluation, usoon_eval, ACC
from sklearn.metrics.cluster import normalized_mutual_info_score
from memory import *
from utils import view_generator
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from consts import ace_structure_test, ace_structure, ere_structure, ere_structure_test
from evaluation_link import *
from link import *
from naming import *

def load_pretrained(args):
    from transformers import BertTokenizer, BertModel, BertConfig
    config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model, config = config)
    generator = view_generator(tokenizer, args.rtr_prob, args.seed)
    return config, tokenizer, bert, generator

def update_centers_l(net, args, known_class_dataloader):
    """
    The code is copied and adapted from https://github.com/Ac-Zyx/RoCORE
    """
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    centers = torch.zeros(args.num_class, args.kmeans_dim, device = device)
    num_samples = [0] * args.num_class
    rep = [[] for i in range(args.num_class)]
    with torch.no_grad():
        for iteration, (input_ids, input_mask, valid_mask, label, pos_span, mask_span, _) in enumerate(known_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, valid_mask, label, pos_span, mask_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, hidden_dim)
            feat = net.forward(data, msg = 'feat').detach()
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label[i]
                rep[l].append(feat[i])
                centers[l] += vec
                num_samples[l] += 1
        for c in range(args.num_class):
            centers[c] /= num_samples[c]
            rep[c] = torch.stack(rep[c], dim = 0)
            assert rep[c].size(0) == num_samples[c]
        net.module.ct_loss_l.centers = centers.to(device)
    return rep

def update_centers_u(net, args, new_class_dataloader):
    """
    The code is copied and adapted from https://github.com/Ac-Zyx/RoCORE
    """
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=args.new_class, random_state=0, algorithm='lloyd')
    true = [-1] * len(new_class_dataloader.dataset)
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        rep = []
        idxes = []
        for iteration, (input_ids, input_mask, valid_mask, label, pos_span, mask_span, idx, _) in enumerate(new_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, valid_mask, label, pos_span, mask_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, kmeans_dim)
            idxes.append(idx)
            rep.append(sia_rep)
        rep = torch.cat(rep, dim = 0).cpu().numpy() # (num_test_ins, kmeans_dim)
        idxes = torch.cat(idxes, dim = 0).cpu().numpy()

    label_pred = clf.fit_predict(rep)# from 0 to args.new_class - 1
    net.module.ct_loss_u.centers = torch.from_numpy(clf.cluster_centers_).to(device) # (num_class, kmeans_dim)
    for i in range(len(idxes)):
        idx = idxes[i]
        pseudo = label_pred[i]
        new_class_dataloader.dataset.examples[idx].pseudo = pseudo
        

def get_neighbor_inds(args, dataloader,model,label):
    memory_bank = MemoryBank(len(dataloader.dataset), args.initial_dim, 0.1)
    fill_memory_bank(dataloader, model, memory_bank, label)
    indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False)
    return indices
    
def get_adjacency(args, inds, neighbors, targets = None, pseudo_label = None, mode = "unlabel"):
    """get adjacency matrix"""
    adj = torch.zeros(inds.shape[0], inds.shape[0])
    ins = 0
    total = 0
    if pseudo_label == None:
        if mode == "unlabel":
            for b1, n in enumerate(neighbors):
                adj[b1][b1] = 1
                for b2, j in enumerate(inds):
                    if j in n:
                        adj[b1][b2] = 1 # if in neighbors
        else :
            for b1, n in enumerate(neighbors):
                adj[b1][b1] = 1
                for b2, j in enumerate(inds):
                    if targets != None and (targets[b1] == targets[b2]) and (targets[b1]>= 0) and (targets[b2]>= 0):
                        adj[b1][b2] = 1 
    
    return adj

def train_one_epoch(net, args, epoch, known_class_dataloader, new_class_dataloader,known_class_dataset, new_class_dataset, optimizer, generator):
    net.train()
    device = torch.device("cuda" if args.cuda else "cpu")

    known_class_iter = iter(known_class_dataloader)
    new_class_iter = iter(new_class_dataloader)
    siamese_known_class_iter = iter(known_class_dataloader)
    siamese_new_class_iter = iter(new_class_dataloader)
    con_known_class_iter = iter(known_class_dataloader)
    con_new_class_iter = iter(new_class_dataloader)

    known_indices, known_distances = get_neighbor_inds(args, known_class_dataloader,net,True)
    new_indices, new_distances = get_neighbor_inds(args,new_class_dataloader,net,False)

    from model import SupConLoss
    criterion = SupConLoss(temperature=0.07, base_temperature=0.07)
    from model import Margin
    M = Margin(args,l_idx2trigger)
    

    epoch_ce_loss = 0
    epoch_kl_loss = 0
    epoch_margin_loss = 0
    epoch_ct_loss = 0
    epoch_rec_loss = 0
    epoch_u_loss = 0
    epoch_l_loss = 0
    epoch_acc = 0 
    total_b = max(len(known_class_dataloader), len(new_class_dataloader))
    with tqdm(total=total_b, desc='training') as pbar:
        for iteration in range(total_b):
            optimizer.zero_grad()
            # net.module.normalize_head()

            # Training unlabeled head
            if epoch > args.num_pretrain:
                data, new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = new_class_iter, batch_size = args.b_size)
                data, pseudo_label = data[:-2], data[-1]
                batch_size = pseudo_label.size(0)

                pseudo_label = (pseudo_label.unsqueeze(0) == pseudo_label.unsqueeze(1)).float().to(device)
                logits = net.forward(data, msg = 'unlabeled') # (batch_size, new_class)

                expanded_logits = logits.expand(batch_size, -1, -1)
                expanded_logits2 = expanded_logits.transpose(0, 1)
                kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
                kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)

                kl_loss = torch.mean(pseudo_label * (kl1 + kl2) + (1 - pseudo_label) * (torch.relu(args.sigmoid - kl1) + torch.relu(args.sigmoid - kl2)))
                kl_loss.backward()
            else:
                kl_loss = torch.tensor(0)

            # Training rec (exclude bert layer)
            data, siamese_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = siamese_known_class_iter, batch_size = args.b_size)
            sia_rep1, rec_loss1 = net.forward(data[:-1], msg = 'reconstruct') # (batch_size, kmeans_dim)
            label = data[3].cuda()

            data, siamese_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = siamese_new_class_iter, batch_size = args.b_size)
            data, pseudo_label = data[:-2], data[-1]
            sia_rep2, rec_loss2 = net.forward(data, msg = 'reconstruct') # (batch_size, kmeans_dim)

            rec_loss = (rec_loss1.mean() + rec_loss2.mean()) / 2
            ct_loss = args.ct * net.module.ct_loss_l(label, sia_rep1)
            ct_loss += args.ct * net.module.ct_loss_u(pseudo_label.cuda(), sia_rep2)

            loss = rec_loss + ct_loss + 1e-5 * (L2Reg(net.module.similarity_encoder) + L2Reg(net.module.similarity_decoder))

            loss.backward()

            if epoch > args.num_pretrain:
                data, known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = known_class_iter, batch_size = args.b_size)
                known_logits = net.forward(data[:-1], msg = 'labeled')

                label_pred = torch.max(known_logits, dim = -1)[1]
                known_label = data[3].to(device)

                acc = 1.0 * torch.sum(label_pred == known_label) / len(label_pred)
                ce_loss = net.module.ce_loss(input = known_logits, target = known_label)
                ce_loss.backward()
            else:
                ce_loss = torch.tensor(0)
                acc = torch.tensor(0)

            # instance level cl
            u_data, con_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = con_new_class_iter, batch_size = args.b_size)
            u_data, idx, pseudo_label = u_data[:-2], u_data[-2], u_data[-1]
            
            neighbor_idx = []
            for indice in new_indices[idx]:
                neighbor_idx.append(np.random.choice(indice, 1)[0])
            u_data2 = new_class_dataset.get_neighbors(neighbor_idx)
            u_data2, idx2, pseudo_label2 = u_data2[:-2], u_data2[-2], u_data2[-1]
            pos_neighbors = torch.from_numpy(new_indices[idx])
            adjacency_u = get_adjacency(args, idx, pos_neighbors, mode ="unlabel")

            u_pos = torch.stack([F.normalize(net.forward(u_data,msg='con'),dim = -1), F.normalize(net.forward(u_data2,msg='con'),dim = -1)], dim=1)
            u_loss = criterion(u_pos, mask=adjacency_u)
            u_loss.backward()

            l_data, con_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = con_known_class_iter, batch_size = args.b_size)
            l_data ,idx = l_data[:-1],l_data[-1]
            neighbor_idx = []
            for indice in known_indices[idx]:
                neighbor_idx.append(np.random.choice(indice, 1)[0])
            l_data2 = known_class_dataset.get_neighbors(neighbor_idx)
            l_data2, idx2 = l_data2[:-1], l_data2[-1]
            pos_neighbors = torch.from_numpy(known_indices[idx])
            adjacency_l = get_adjacency(args, idx, pos_neighbors, l_data[3], mode = "label")

            l_pos = torch.stack([F.normalize(net.forward(l_data,msg='con'),dim=-1), F.normalize(net.forward(l_data2,msg='con'),dim=-1)], dim=1)
            l_loss = 0.3 * criterion(l_pos, mask=adjacency_l)
            l_loss.backward()
            
            if epoch > args.warmup:
                pos_data = known_class_dataset.get_pos(l_data[3])
                pos_data, pos_idx = pos_data[:-1], pos_data[-1]
                p_label = pos_data[3]
                
                neg_data = known_class_dataset.get_neg(l_data[3])
                neg_data, neg_idx = neg_data[:-1], neg_data[-1]
                n_label = neg_data[3]

                margin = M.get_margin(p_label,n_label)

                margin = torch.LongTensor(margin).to(net.module.device)
                pos_logits = net.forward(pos_data, msg="margin")
                neg_logits = net.forward(neg_data, msg="margin")

                logits = net.forward(l_data, msg="margin")
                pos_score = (pos_logits * logits).sum(dim = -1)
                neg_score = (neg_logits * logits).sum(dim = -1)
                margin_loss = args.margin_weight * (-pos_score + neg_score + margin).clamp(min=0).sum()
                margin_loss.backward()
            else:
                margin_loss = torch.tensor(0)
            
            
            optimizer.step()  
            epoch_kl_loss += kl_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_ct_loss += ct_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_u_loss += u_loss.item()
            epoch_l_loss += l_loss.item()
            epoch_margin_loss += margin_loss.item()
            epoch_acc += acc.item()
            pbar.update(1)
            pbar.set_postfix({"acc":epoch_acc / (iteration + 1), "margin loss":epoch_margin_loss / (iteration + 1),"kl loss":epoch_kl_loss / (iteration + 1),"ce loss":epoch_ce_loss / (iteration + 1), "rec loss":epoch_rec_loss / (iteration + 1), "ct_loss":epoch_ct_loss / (iteration + 1), "u_loss":epoch_u_loss / (iteration + 1), "label_loss":epoch_l_loss / (iteration + 1), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
    num_iteration = total_b
    print("===> Epoch {} Complete: Avg. ce Loss: {:.4f}, rec Loss: {:.4f}, ct Loss: {:.4f}, u Loss: {:.4f}, known class acc: {:.4f}".format(epoch, epoch_ce_loss / num_iteration, epoch_rec_loss / num_iteration, epoch_ct_loss / num_iteration, epoch_u_loss / num_iteration, epoch_acc / num_iteration))
    

def test_one_epoch_ulabel(net, args, epoch, new_class_dataloader):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        with tqdm(total=len(new_class_dataloader), desc='testing') as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                p_label = data[-1]
                data = data[:-2]
                logits = net.forward(data, msg = 'unlabeled')
                ground_truth.append(data[3])
                label_pred.append(logits.max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy() 
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
            B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
            a = ACC(ground_truth, label_pred)
            print("acc:{}, B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}, NMI:{}".format(a, B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI,normalized_mutual_info_score(ground_truth, label_pred)))
            print(cluster_eval)
    return cluster_eval['F1'], ARI

def test_one_epoch_label(net, args, epoch, new_class_dataloader):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        with tqdm(total=len(new_class_dataloader), desc="label") as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                logits = net.forward(data[:-1], msg = 'labeled')
                ground_truth.append(data[3])
                label_pred.append(logits.max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy()
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
    print(cluster_eval)
    return cluster_eval['F1']

def save_tsne_pic(net, args, epoch, new_class_dataloader):
    from utils import tsne_and_save_pic
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    with torch.no_grad():
        ground_truth = []
        vec = []
        with tqdm(total=len(new_class_dataloader), desc="pca") as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                feat = net.forward(data[:-2], msg = 'feat')
                ground_truth.append(data[3])
                vec.append(feat)
                pbar.update(1)
            feat_vec = torch.cat(vec, dim = 0).cpu().numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).cpu().numpy()
    tsne_and_save_pic(feat_vec, ground_truth ,True, epoch)

def main(args):
    config, tokenizer, pretrained_model, generator = load_pretrained(args)
    # get data
    if args.dataset == 'ace':
        from consts import LABEL_TRIGGERS_ACE,UNLABEL_TRIGGERS_ACE
        known_class_train_examples = BertDataset.preprocess(args.root + args.known_class_filename, LABEL_TRIGGERS_ACE, l_trigger2idx)
        new_class_train_examples = PBertDataset.preprocess(args.root + args.new_class_filename, UNLABEL_TRIGGERS_ACE, u_trigger2idx)
        known_class_test_examples = BertDataset.preprocess(args.root + args.test_class_filename, LABEL_TRIGGERS_ACE, l_trigger2idx)
        new_class_test_examples = PBertDataset.preprocess(args.root + args.test_class_filename, UNLABEL_TRIGGERS_ACE, u_trigger2idx)
    elif args.dataset == 'ere':
        from consts import LABEL_TRIGGERS_ERE,UNLABEL_TRIGGERS_ERE
        known_class_train_examples = BertDataset.preprocess_ere(args.root + args.known_class_filename, LABEL_TRIGGERS_ERE,l_trigger2idx)
        new_class_train_examples = PBertDataset.preprocess_ere(args.root + args.new_class_filename, UNLABEL_TRIGGERS_ERE,u_trigger2idx)
        known_class_test_examples = BertDataset.preprocess_ere(args.root + args.test_class_filename, LABEL_TRIGGERS_ERE,l_trigger2idx)
        new_class_test_examples = PBertDataset.preprocess_ere(args.root + args.test_class_filename, UNLABEL_TRIGGERS_ERE,u_trigger2idx)
    elif args.dataset == 'maven':
        from consts import LABEL_TRIGGERS_MAVEN,UNLABEL_TRIGGERS_MAVEN
        known_class_train_examples = BertDataset.preprocess_maven(args.root + args.known_class_filename, LABEL_TRIGGERS_MAVEN,l_trigger2idx)
        new_class_train_examples = PBertDataset.preprocess_maven(args.root + args.new_class_filename, UNLABEL_TRIGGERS_MAVEN,u_trigger2idx)
        known_class_test_examples = BertDataset.preprocess_maven(args.root + args.test_class_filename, LABEL_TRIGGERS_MAVEN,l_trigger2idx)
        new_class_test_examples = PBertDataset.preprocess_maven(args.root + args.test_class_filename, UNLABEL_TRIGGERS_MAVEN,u_trigger2idx)
    
    # get dataset and dataloader
    known_class_dataset = BertDataset(args, known_class_train_examples, tokenizer)
    known_class_train_dataloader = DataLoader(known_class_dataset, batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    known_class_test_dataloader = DataLoader(BertDataset(args, known_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("knwon class dataloader ready...")
    new_calss_dataset = PBertDataset(args, new_class_train_examples, tokenizer)
    new_class_train_dataloader = DataLoader(new_calss_dataset, batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    new_class_test_dataloader = DataLoader(PBertDataset(args, new_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("new class dataloader ready...")

    net = HALTON(args, config, pretrained_model, unfreeze_layers = [args.layer])
    if args.cuda:
        net = nn.DataParallel(net, device_ids = [0,1]).cuda()

    optimizer = optim.Adam(net.parameters(), lr = args.lr)

    best_result = 0
    best_test_result = 0
    wait_times = 0


    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        if epoch > args.num_pretrain:
            wait_times += 1
        update_centers_u(net, args, new_class_train_dataloader)
        rep = update_centers_l(net, args, known_class_train_dataloader)
        train_one_epoch(net, args, epoch, known_class_train_dataloader, new_class_train_dataloader, known_class_dataset, new_calss_dataset, optimizer, generator)
        
        test_one_epoch_label(net, args, epoch, known_class_test_dataloader)
        update_centers_u(net, args, new_class_test_dataloader)
        _, result = test_one_epoch_ulabel(net, args, epoch, new_class_train_dataloader)
        _, test_result = test_one_epoch_ulabel(net, args, epoch, new_class_test_dataloader)
        
        if result > best_result:
            wait_times = 0
            best_result = result
            best_test_result = test_result
            print("new class dev best result: {}, test result: {}".format(best_result, test_result))

            if args.dataset == "ace":
                torch.save(net, args.save+"model_ace_"+ str(args.seed) +"final.pt")
            elif args.dataset == "ere":
                torch.save(net, args.save+"model_ere_"+ str(args.seed) +"final.pt")
            elif args.dataset == "maven":
                torch.save(net, args.save+"model_maven_"+ str(args.seed) +"final.pt")
            # link
            
                
        if wait_times > args.wait_times or epoch == args.epochs:
            print("wait times arrive: {}, stop training, best result is: {}".format(args.wait_times, best_test_result))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Bert probe task for entity extraction')
    parser.add_argument("--load", type = str)
    parser.add_argument("--save", type = str, default = './model/')
    parser.add_argument("--dataset", type = str, choices = ['ace', 'ere', 'maven'])
    parser.add_argument("--rel_filename", type = str, default = "relation_description.txt")
    parser.add_argument("--known_class_filename", type = str, default = "train.json")
    parser.add_argument("--new_class_filename", type = str, default = "train.json")
    parser.add_argument("--test_class_filename", type = str, default = "test_dev.json")
    parser.add_argument("--root", type = str, default = "../data/ace/")
    parser.add_argument("--mode", type = int, default = 3)
    parser.add_argument("--p", type = float, default=1.0)
    parser.add_argument("--layer", type = int, default = 12)
    parser.add_argument("--b_size", type = int, default = 128)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--max_len", type = int, default = 160)
    parser.add_argument("--epochs", type = int, default = 60)
    parser.add_argument("--wait_times", type = int, default = 10)
    parser.add_argument("--rtr_prob",type = int,default = 0.25)
    parser.add_argument("--ct", type = float, default = 0.005)
    parser.add_argument("--num_pretrain", type = float, default = 0)
    parser.add_argument("--sigmoid", type = float, default = 2)
    parser.add_argument("--initial_dim",type=int, default=768)
    parser.add_argument("--hidden_dim", type = int, default = 512)
    parser.add_argument("--kmeans_dim", type = int, default = 256)
    parser.add_argument("--num_class", type = int, default = 10)
    parser.add_argument("--new_class", type = int, default = 23)
    parser.add_argument("--margin_weight", type=float, default = 0.05)
    parser.add_argument("--topk", type = int , default = 2)
    parser.add_argument("--cuda", action = 'store_true', help = 'use CUDA')
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--view_strategy",type = str,default = "none")
    parser.add_argument("--taxo_path", type = str, default= "./taxonomy/ace/ace.taxo")
    parser.add_argument("--use_graph",type=bool,default=False)
    parser.add_argument("--naming",type=bool,default=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--bert_model", 
                          default="bert-base-uncased", 
                          type=str,
                          help="bert pre-trained model selected in the list: bert-base-uncased, "
                          "bert-large-uncased, bert-base-cased, bert-large-cased")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

        
    seed_everything(args.seed)
    if args.dataset == "ace":
        from consts import LABEL_TRIGGERS_ACE,UNLABEL_TRIGGERS_ACE
        all_l_triggers, l_trigger2idx, l_idx2trigger = build_vocab(LABEL_TRIGGERS_ACE)
        all_u_triggers, u_trigger2idx, u_idx2trigger = build_vocab(UNLABEL_TRIGGERS_ACE)
    if args.dataset == 'ere':
        args.root = '../data/ere/'
        args.known_class_filename = 'train.json'
        args.new_class_filename = 'train.json'
        args.test_class_filename = 'test_dev.json'
        args.taxo_path = "./taxonomy/ere/ere.taxo"
        from consts import LABEL_TRIGGERS_ERE,UNLABEL_TRIGGERS_ERE
        all_l_triggers, l_trigger2idx, l_idx2trigger = build_vocab(LABEL_TRIGGERS_ERE)
        all_u_triggers, u_trigger2idx, u_idx2trigger = build_vocab(UNLABEL_TRIGGERS_ERE)
        args.num_class = len(l_trigger2idx)
        args.new_class = len(u_trigger2idx)
    elif args.dataset == 'maven':
        args.root = '../data/maven/'
        args.known_class_filename = 'train.jsonl'
        args.new_class_filename = 'train.jsonl'
        args.test_class_filename = 'valid.jsonl'
        args.taxo_path = "./taxonomy/maven/maven.taxo"
        args.topk = 15
        args.b_size = 256

        from consts import LABEL_TRIGGERS_MAVEN,UNLABEL_TRIGGERS_MAVEN
        all_l_triggers, l_trigger2idx, l_idx2trigger = build_vocab(LABEL_TRIGGERS_MAVEN)
        all_u_triggers, u_trigger2idx, u_idx2trigger = build_vocab(UNLABEL_TRIGGERS_MAVEN)
        args.num_class = len(l_trigger2idx)
        args.new_class = len(u_trigger2idx) 
    print(args)
    main(args)