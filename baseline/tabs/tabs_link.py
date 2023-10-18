import sys
path = "../../../HALTON/"
sys.path.append(path)

import math
import numpy as np
import json
from sklearn import metrics
from copy import copy
import torch

class HierarchyCluster():
    def __init__(self, rel_wiki_id: str = "", instances: list = None, sons: list = None, fathers: list = None,
                 degree: float = 0, rel_type=""):
        self.rel_wiki_id = rel_wiki_id
        self.instances = instances if instances else []
        self.sons = sons if sons else []
        self.fathers = fathers if fathers else []
        self.degree = degree  # inner connection
        self.rel_type = rel_type
        self.insert_paths = []  # contains tuple (father, avg_link score)

    def __copy__(self):
        return HierarchyCluster(self.rel_wiki_id, self.instances.copy(), self.sons.copy(), self.fathers.copy(),
                                self.degree, self.rel_type)

def create_info(args, structure):
    if args.dataset == "ace":
        info = {}
        for item in structure:
            if item[1][0] not in info.keys():
                info[item[1][0]] = {"sons":[],"fathers":[]}
            if item[0][0] != "Business":
                info[item[1][0]]['fathers'].append(item[0][0])
        return info      
    elif args.dataset == "ere":
        info = {}
        for item in structure:
            if item[1][0] not in info.keys():
                info[item[1][0]] = {"sons":[],"fathers":[]}
            if item[0][0] != "Business" and item[0][0] != "Manufacture":
                info[item[1][0]]['fathers'].append(item[0][0])
        return info   
    elif args.dataset == "maven":
        info = {}
        for item in structure:
            if item[1][0] not in info.keys():
                info[item[1][0]] = {"sons":[],"fathers":[]}
            if item[0][0] == "event_type":
                continue
            else:
                info[item[1][0]]['fathers']= item[0][0].split(":")
        return info   

def get_glod_hierarchy_cluster_list(info, dataloader, dict):
    gold_hierarchy_cluster_list = []
    instances = [[] for i in range(len(info))]
    assert len(info) == len(dict)
    for iteration, (input_ids, input_mask, valid_mask, label, pos_span, mask_span, idx, _, _) in enumerate(dataloader): # (batch_size, seq_len), (batch_size)
        for i in range(len(label)):
            instances[label[i].item()].append(idx[i].item())
    for rel_wiki_id in info.keys():
        sons = info[rel_wiki_id]['sons']
        fathers = info[rel_wiki_id]["fathers"]
        instance = instances[dict[rel_wiki_id]]
        gold_hierarchy_cluster_list.append(HierarchyCluster(rel_wiki_id, instance, sons, fathers))
    return gold_hierarchy_cluster_list

def get_predict_cluster_list(res):
    predicted_cluster_list = []
    for rel_wiki_id in res.keys():
        sons = res[rel_wiki_id]['sons']
        fathers = res[rel_wiki_id]["fathers"]
        instance = res[rel_wiki_id]["instance"]
        predicted_cluster_list.append(HierarchyCluster(rel_wiki_id, instance, sons, fathers))
    return predicted_cluster_list

def get_test_info(args, dataloader, dict, net):
    info = [{"name":None, "emb": None, "instance":[], "vec":[], "name_LLM":None} for i in dict]
    net.eval()
    device=torch.device("cuda" if args.cuda else "cpu")

    with torch.no_grad():
        index = []
        label_pred = []
        label = []
        centers = torch.zeros(args.new_class, 256, device=device)
        for iteration, data in enumerate(dataloader):
            idx = data[-3]

            logits = net.module.encoder(data)
            sia_rep, v2 = net.module.get_cluster_feat(data)
            sia_rep = sia_rep + v2
            
            label_pred = logits.max(dim = -1)[1].cpu()
            assert len(sia_rep) == len(label_pred)
            assert len(idx) == len(label_pred)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label_pred[i]
                centers[l] += vec
                info[l]["name"] = l.item()
                info[l]["instance"].append(idx[i].item())
                info[l]["vec"].append(vec)
        for i in range(len(info)):
            if len(info[i]['vec']) > 0:
                info[i]['vec'] = torch.stack(info[i]['vec'], dim=0)
                assert info[i]['vec'].size(0) == len(info[i]["instance"])
        for c in range(args.new_class):
            if len(info[c]["instance"]) > 0:
                info[c]["emb"] = centers[c] / len(info[c]["instance"])
            else:
                info[c]["emb"] = centers[c]
    return info

def get_gold_info(args, dataloader, dict, net):
    info = [{"name":None, "emb": None, "instance":[],"vec":[],"name_LLM":None} for i in dict]
    net.eval()
    device=torch.device("cuda" if args.cuda else "cpu")
    from consts import UNLABEL_TRIGGERS_ACE, UNLABEL_TRIGGERS_ERE, naming_unlabel_trigger_ace, naming_unlabel_trigger_ere, UNLABEL_TRIGGERS_MAVEN
    if args.dataset == "ace":
        trigger_name_dict = UNLABEL_TRIGGERS_ACE
        trigger_dict = UNLABEL_TRIGGERS_ACE
    elif args.dataset == "ere":
        trigger_name_dict = UNLABEL_TRIGGERS_ERE
        trigger_dict = UNLABEL_TRIGGERS_ERE
    elif args.dataset == "maven":
        trigger_name_dict = UNLABEL_TRIGGERS_MAVEN
        trigger_dict = UNLABEL_TRIGGERS_MAVEN

    with torch.no_grad():
        index = []
        label_pred = []
        label = []
        centers = torch.zeros(args.new_class, 256, device=device)
        for iteration, data in enumerate(dataloader):
            idx = data[-3]
            label_gold = data[3].cpu()
            # logits = net.forward(data, msg = 'unlabeled')
            # label_pred = logits.max(dim = -1)[1].cpu()
            sia_rep, v2 = net.module.get_cluster_feat(data)
            sia_rep = sia_rep + v2

            assert len(sia_rep) == len(label_gold)
            assert len(idx) == len(label_gold)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label_gold[i]
                centers[l] += vec
                info[l]["name"] = l.item()
                info[l]["instance"].append(idx[i].item())
                info[l]["vec"].append(vec)
        for i in range(len(info)):
            if len(info[i]['vec']) > 0:
                info[i]['vec'] = torch.stack(info[i]['vec'], dim=0)
                assert info[i]['vec'].size(0) == len(info[i]["instance"])
        for c in range(args.new_class):
            if len(info[c]["instance"]) > 0:
                info[c]["emb"] = centers[c] / len(info[c]["instance"])
            else:
                info[c]["emb"] = centers[c]
            info[c]["name_LLM"] = [trigger_name_dict[c]]
    return info

def get_tree_info(net, args, known_class_dataloader):
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    num_samples = [0] * args.num_class
    rep = [[] for i in range(args.num_class)]
    with torch.no_grad():
        for iteration, data in enumerate(known_class_dataloader): # (batch_size, seq_len), (batch_size)
            label = data[3]
            feat,v2 = net.module.get_cluster_feat(data, True)
            feat = feat + v2
            for i in range(len(feat)):
                l = label[i]
                rep[l].append(feat[i].detach())
                num_samples[l] += 1
        for c in range(args.num_class):
            rep[c] = torch.stack(rep[c], dim = 0)
            assert rep[c].size(0) == num_samples[c]
    return rep


class HierarchyClusterEvaluation:
    def __init__(self, gt_cluster_list, predicted_cluster_list, test_data_num):
        """unduplicated match"""
        self.gt_cluster_list = gt_cluster_list
        self.predicted_cluster_list = predicted_cluster_list
        self.relation_ground_dict = dict()  # ground the predicted relation cluster to gt relation cluster
        self.reverse_relation_ground_dict = dict()
        # to avoid same and easy calculate precision and recall (set operation)
        for index, predicted_cluster in enumerate(predicted_cluster_list):
            self.relation_ground_dict[predicted_cluster.rel_wiki_id] = 'Not grounded' + str(index)
        for index, gt_cluster in enumerate(gt_cluster_list):
            self.reverse_relation_ground_dict[gt_cluster.rel_wiki_id] = 'Not grounded' + str(index)
        # predicted and gt element num are same!
        self.all_element_num = test_data_num

    def get_relation(self):
        return self.relation_ground_dict

    def match_all_predicted_cluster(self):
        all_match_f1 = np.zeros((len(self.predicted_cluster_list), len(self.gt_cluster_list)))
        for p_i, p_c in enumerate(self.predicted_cluster_list):
            for g_i, g_c in enumerate(self.gt_cluster_list):
                p = self.precision(set(p_c.instances), set(g_c.instances))
                r = self.recall(set(p_c.instances), set(g_c.instances))
                match_f1 = 2 * r * p / (p + r) if p + r > 0 else 0
                all_match_f1[p_i, g_i] = match_f1
        # print("*"*64+"match_f1")
        # print(all_match_f1)
        for i in range(len(self.predicted_cluster_list)):
            if np.max(all_match_f1) <= 0:  # all matched
                break
            row_i = np.argmax(all_match_f1) // len(self.gt_cluster_list)
            col_i = np.argmax(all_match_f1) % len(self.gt_cluster_list)
            p_c = self.predicted_cluster_list[row_i]
            g_c = self.gt_cluster_list[col_i]
            # set match
            self.relation_ground_dict[p_c.rel_wiki_id] = g_c.rel_wiki_id
            self.reverse_relation_ground_dict[g_c.rel_wiki_id] = p_c.rel_wiki_id
            # set zeros after matched
            # print("*"*64+"match_f1 row{} col{} max{} pos{}".format(row_i, col_i, np.max(all_match_f1), np.argmax(all_match_f1)))

            all_match_f1[row_i, :] = -1
            all_match_f1[:, col_i] = -1
            # print(all_match_f1)
        return

    def precision(self, response_a, reference_a):
        if len(response_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(response_a))

    def recall(self, response_a, reference_a):
        if len(reference_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def TotalElementR_P(self):
        totalRecall = 0.0
        totalPrecision = 0.0
        taxonomyRecall = 0.0
        taxonomyPrecision = 0.0

        f1_dict = dict()  # record the cluster F1 score to avoid more calculation.
        reversed_relation_grounded_dict = dict()

        # calculate TP_{sc}
        for predicted_cluster in self.predicted_cluster_list:
            predicted_cluster_instances = set(predicted_cluster.instances)
            predicted_taxonomy_nodes = set(predicted_cluster.sons + predicted_cluster.fathers)
            matched_gt_key = self.relation_ground_dict[predicted_cluster.rel_wiki_id]
            if not matched_gt_key.startswith('Not grounded'):
                for gt_cluster in self.gt_cluster_list:
                    if gt_cluster.rel_wiki_id == matched_gt_key:
                        matched_gt_cluster = gt_cluster
                gt_matched_cluster_instances = set(matched_gt_cluster.instances)
                gt_taxonomy_nodes = set(matched_gt_cluster.sons + matched_gt_cluster.fathers)
            else:
                gt_matched_cluster_instances = set()
                gt_taxonomy_nodes = set()

            # calculate the cluster
            cluster_recall = self.recall(predicted_cluster_instances, gt_matched_cluster_instances)
            cluster_precision = self.precision(predicted_cluster_instances, gt_matched_cluster_instances)
            if cluster_precision + cluster_recall > 0:
                cluster_f1 = 2 * cluster_precision * cluster_recall / (cluster_precision + cluster_recall)
            else:
                cluster_f1 = 0
            # calculate the taxonomy
            if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0 and not matched_gt_key.startswith(
                    'Not grounded'):
                taxonomy_precision = 1
            else:
                taxonomy_precision = self.precision(predicted_taxonomy_nodes, gt_taxonomy_nodes)
                

            taxonomyPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision
            # combine above.
            # print(1 / len(self.predicted_cluster_list) * taxonomy_precision)

            totalPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision * cluster_f1
            # record information for recall calculation.
            if not matched_gt_key.startswith('Not grounded'):
                f1_dict[matched_gt_key] = cluster_f1
                reversed_relation_grounded_dict[matched_gt_key] = predicted_cluster.rel_wiki_id

        # calculate TR_{sc}

        for gt_cluster in self.gt_cluster_list:
            if gt_cluster.rel_wiki_id in reversed_relation_grounded_dict.keys():  # grounded.
                cluster_f1 = f1_dict[gt_cluster.rel_wiki_id]
                predicted_cluster_id = reversed_relation_grounded_dict[gt_cluster.rel_wiki_id]
                for predicted_cluster in self.predicted_cluster_list:
                    if predicted_cluster.rel_wiki_id == predicted_cluster_id:
                        matched_predicted_cluster = predicted_cluster
                        break
                gt_taxonomy_nodes = set(gt_cluster.sons + gt_cluster.fathers)
                predicted_taxonomy_nodes = set(matched_predicted_cluster.sons + matched_predicted_cluster.fathers)
                if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0:
                    taxonomy_recall = 1
                else:
                    taxonomy_recall = self.recall(predicted_taxonomy_nodes, gt_taxonomy_nodes)
                    # print("recall name {} predict {} true {}".format(gt_cluster.rel_wiki_id, predicted_taxonomy_nodes, gt_taxonomy_nodes))
            else:
                cluster_f1 = 0
                taxonomy_recall = 0

            taxonomyRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall
            totalRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall * cluster_f1
        return taxonomyRecall, taxonomyPrecision, totalRecall, totalPrecision
    
    def MatchF1(self):
        totalF1 = 0.0
        totalP = 0.0
        totalR = 0.0
        for p_i, p_c in enumerate(self.predicted_cluster_list):
            g_c_id = self.relation_ground_dict[p_c.rel_wiki_id]
            if g_c_id.startswith("Not grounded"):
                continue
            else:
                for g_c in self.gt_cluster_list:
                    if g_c_id == g_c.rel_wiki_id:
                        p = self.precision(set(p_c.instances), set(g_c.instances))
                        r = self.recall(set(p_c.instances), set(g_c.instances))
                        match_f1 = 2 * r * p / (p + r) if p + r > 0 else 0
                        # print("p_c_name{} g_c_name{} match F1{} p{} r{}".format(p_c.rel_wiki_id, g_c.rel_wiki_id, match_f1, p, r))
                        totalF1 += match_f1 * (len(p_c.instances) / self.all_element_num)
                        totalP += p * (len(p_c.instances) / self.all_element_num)
                        totalR += r * (len(p_c.instances) / self.all_element_num)
        return totalF1, totalP, totalR

    def printEvaluation(self, print_flag=True):
        self.match_all_predicted_cluster()
        match_f1, match_P, match_R = self.MatchF1()
        
        taxonomy_rec, taxonomy_prec, total_rec, total_prec = self.TotalElementR_P()
        if total_rec == 0 and total_prec == 0:
            total_f1 = 0
        else:
            total_f1 = (2 * total_rec * total_prec) / (total_rec + total_prec)
        if taxonomy_rec == 0 and taxonomy_prec == 0:
            taxonomy_f1 = 0
        else:
            taxonomy_f1 = (2 * taxonomy_rec * taxonomy_prec) / (taxonomy_rec + taxonomy_prec)
        if print_flag:
            print("new metric Info:")
            print("Precision(%); Recall(%); F1(%)")
            print(round(match_P * 100, 3), "; ", round(match_R * 100, 3), "; ", round(match_f1 * 100, 3))
            

            print("taxonomy Info:")
            print("Precision(%); Recall(%); F1(%)")
            print(round(taxonomy_prec * 100, 3), "; ", round(taxonomy_rec * 100, 3), "; ", round(taxonomy_f1 * 100, 3))

            print("Total Info:")
            print("Precision(%); Recall(%); F1(%)")
            print(round(total_prec * 100, 3), "; ", round(total_rec * 100, 3), "; ", round(total_f1 * 100, 3))

        m = {'match_f1': match_f1, 'total_F1': total_f1, 'total_precision': total_prec,
             'total_recall': total_rec,
             'taxonomy_F1': taxonomy_f1, 'taxonomy_precision': taxonomy_prec, 'taxonomy_recall': taxonomy_rec}
        # m = {k: v * 100 for k, v in m.items()}
        return m