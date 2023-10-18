import sys
path = "../../../mask_view/"
sys.path.append(path)

from dataset import Labeled_Dataset as BertDataset
from dataset import unLabeled_Dataset as PBertDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from vqvae_model import *
from utils import seed_everything, data_split2, L2Reg, compute_kld, _worker_init_fn_,build_vocab, endless_get_next_batch
import random
import os
from evaluation import ClusterEvaluation, usoon_eval, ACC
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import view_generator
from sentence_transformers import SentenceTransformer, util
from consts import ace_structure_test, ace_structure, ere_structure, ere_structure_test
from vqvae_link import *
from link import *

# 先训练AUTOENCODER 10轮，再一起训
def load_pretrained(args):
    from transformers import BertTokenizer, BertModel, BertConfig
    config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model, config = config)
    generator = view_generator(tokenizer, args.rtr_prob, args.seed)
    return config, tokenizer, bert, generator

def train_one_epoch(net, args, epoch, known_class_dataloader, new_class_dataloader,known_class_dataset, new_class_dataset, optimizer, generator):
    net.train()
    device = torch.device("cuda" if args.cuda else "cpu")
    known_class_iter = iter(known_class_dataloader)
    new_class_iter = iter(new_class_dataloader)

    epoch_loss = 0

    with tqdm(total=len(new_class_dataloader), desc='training') as pbar:
        for iteration in range(len(new_class_dataloader)):
            optimizer.zero_grad()

            # Training unlabeled head
          
            # Training unlabeled head
            data, new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = new_class_iter, batch_size = args.b_size)
            u_data, pseudo_label = data[:-2], data[-1]
            u_loss = net(u_data, is_l=False)
            data, known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = known_class_iter, batch_size = args.b_size)
            l_data = data[:-1]
            l_loss = net(l_data)
            loss = l_loss + u_loss
            loss.backward()

            optimizer.step()  
            epoch_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({"loss":epoch_loss / (iteration + 1),'lr': optimizer.state_dict()['param_groups'][0]['lr']})
    num_iteration = len(new_class_dataloader)
    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / num_iteration, epoch_rec_loss / num_iteration, epoch_ct_loss / num_iteration, epoch_u_loss / num_iteration, epoch_acc / num_iteration))
    

def test_one_epoch(net, args, epoch, new_class_dataloader):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        pseudos = []
        with tqdm(total=len(new_class_dataloader), desc='testing') as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                data = data[:-2]
                logits = net.module.encoder(data)
                ground_truth.append(data[3])
                label_pred.append(logits[:, args.num_class:].max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy() 
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
            B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
            a = ACC(ground_truth, label_pred)
            print("acc:{}, B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}, NMI:{}".format(a, B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI,normalized_mutual_info_score(ground_truth, label_pred)))
            print(cluster_eval)
    return cluster_eval['F1'],ARI

def test_one_epoch2(net, args, epoch, new_class_dataloader, labelled = False):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    desc = 'labelled' if labelled else 'unlabelled'
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        with tqdm(total=len(new_class_dataloader), desc=desc) as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                if not labelled:
                    data = data[:-2]
                logits = net.module.encoder(data[:-1])
                ground_truth.append(data[3])
                label_pred.append(logits[:, args.num_class:].max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy()
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
    print(cluster_eval)
    return cluster_eval['F1'], label_pred, ground_truth

def main(args):
    config, tokenizer, pretrained_model, generator = load_pretrained(args)

    if args.dataset == 'ace':
        from consts import LABEL_TRIGGERS_ACE,UNLABEL_TRIGGERS_ACE
        known_class_train_examples = BertDataset.preprocess(args.root + args.known_class_filename, LABEL_TRIGGERS_ACE,l_trigger2idx)
        new_class_train_examples = PBertDataset.preprocess(args.root + args.new_class_filename, UNLABEL_TRIGGERS_ACE,u_trigger2idx)
        known_class_test_examples = BertDataset.preprocess(args.root + args.test_class_filename, LABEL_TRIGGERS_ACE,l_trigger2idx)
        new_class_test_examples = PBertDataset.preprocess(args.root + args.test_class_filename, UNLABEL_TRIGGERS_ACE,u_trigger2idx)
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
    known_class_dataset = BertDataset(args, known_class_train_examples, tokenizer)
    known_class_train_dataloader = DataLoader(known_class_dataset, batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    known_class_test_dataloader = DataLoader(BertDataset(args, known_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("knwon class dataloader ready...")
    new_calss_dataset = PBertDataset(args, new_class_train_examples, tokenizer)
    new_class_train_dataloader = DataLoader(new_calss_dataset, batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    new_class_test_dataloader = DataLoader(PBertDataset(args, new_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("new class dataloader ready...")
    net = VQVAEModel(args, tokenizer)
    if args.cuda:
        net = nn.DataParallel(net, device_ids = [0]).cuda()
    print("net ready...")
    print("-"*32)
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    best_result = 0
    best_test_result = 0
    wait_times = 0
    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        if epoch > args.num_pretrain:
            wait_times += 1

        train_one_epoch(net, args, epoch, known_class_train_dataloader, new_class_train_dataloader,known_class_dataset,new_calss_dataset, optimizer,generator)

        test_one_epoch2(net, args, epoch, known_class_test_dataloader, labelled = True)
        _,result = test_one_epoch(net, args, epoch, new_class_train_dataloader)
        _,test_result = test_one_epoch(net, args, epoch, new_class_test_dataloader)
        
        if result > best_result:
            wait_times = 0
            best_result = result
            best_test_result = test_result
            print("new class dev best result: {}, test result: {}".format(best_result, test_result))
            if args.dataset == "ace":
                torch.save(net, args.save+"model_ssvqvae_ace_"+ str(args.seed) +".pt")
            elif args.dataset == "ere":
                torch.save(net, args.save+"model_ssvqvae_ere_"+ str(args.seed) +".pt")
            elif args.dataset == "maven":
                torch.save(net, args.save+"model_ssvqvae_maven_"+ str(args.seed) +".pt")

            
        if wait_times > args.wait_times:
            break
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Bert probe task for entity extraction')
    parser.add_argument("--dataset", type = str, choices = ['ace', 'ere', 'maven'])
    parser.add_argument("--save", type = str, default = './model/')
    parser.add_argument("--known_class_filename", type = str, default = "train.json")
    parser.add_argument("--new_class_filename", type = str, default = "train.json")
    parser.add_argument("--test_class_filename", type = str, default = "test_dev.json")
    parser.add_argument("--root", type = str, default = "../../../data/ace/")
    parser.add_argument("--p", type = float, default=1.0)
    parser.add_argument("--layer", type = int, default = 12)
    parser.add_argument("--b_size", type = int, default = 12)
    parser.add_argument("--clip_grad", type = float, default = 0)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--max_len", type = int, default = 160)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--wait_times", type = int, default = 5)
    parser.add_argument("--rtr_prob",type = int,default = 0.25)
    parser.add_argument("--ct", type = float, default = 0.005)
    parser.add_argument("--num_pretrain", type = float, default = 0)
    parser.add_argument("--sigmoid", type = float, default = 2)
    parser.add_argument("--hidden_dim", type = int, default = 512)
    parser.add_argument("--kmeans_dim", type = int, default = 256)
    parser.add_argument("--num_class", type = int, default = 10)
    parser.add_argument("--new_class", type = int, default = 23)
    parser.add_argument("--cuda", action = 'store_true', help = 'use CUDA')
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--beta", type = float, default = 1)
    parser.add_argument("--gamma", type = float, default = 0)
    parser.add_argument("--use_graph", type=bool, default=False)
    parser.add_argument("--recon_loss", type = float, default = 0)
    parser.add_argument("--view_strategy",type = str,default = "none")
    parser.add_argument("--bert_model", 
                          default="bert-base-uncased", 
                          type=str,
                          help="bert pre-trained model selected in the list: bert-base-uncased, "
                          "bert-large-uncased, bert-base-cased, bert-large-cased")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    seed_everything(args.seed)
    if args.dataset == "ace":
        from consts import LABEL_TRIGGERS_ACE,UNLABEL_TRIGGERS_ACE
        all_l_triggers, l_trigger2idx, l_idx2trigger = build_vocab(LABEL_TRIGGERS_ACE)
        all_u_triggers, u_trigger2idx, u_idx2trigger = build_vocab(UNLABEL_TRIGGERS_ACE)
    if args.dataset == 'ere':
        args.root = '../../../data/ere/'
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
        args.root = '../../../data/maven/'
        args.known_class_filename = 'train.jsonl'
        args.new_class_filename = 'train.jsonl'
        args.test_class_filename = 'valid.jsonl'
        
        from consts import LABEL_TRIGGERS_MAVEN,UNLABEL_TRIGGERS_MAVEN
        all_l_triggers, l_trigger2idx, l_idx2trigger = build_vocab(LABEL_TRIGGERS_MAVEN)
        all_u_triggers, u_trigger2idx, u_idx2trigger = build_vocab(UNLABEL_TRIGGERS_MAVEN)
        args.num_class = len(l_trigger2idx)
        args.new_class = len(u_trigger2idx)
    print(args)
    main(args)