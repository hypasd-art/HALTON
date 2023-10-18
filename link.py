import torch
import torch.nn as nn 
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
from evaluation_link import HierarchyClusterEvaluation
from utils import build_vocab
import math
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn.functional as F
class node:
    def __init__(self, name, emb = 0):
        self.child = []
        self.parent = None
        self.emb = emb
        self.name = name
        self.ancestors = []
        self.height = 1
        self.name_vec = None
        self.rep = None
        
from nltk.corpus import wordnet as wn
def calcu_path_sim(word1, word2):
    w1= wn.synsets(word1)
    w2 = wn.synsets(word2)
    score = 0
    if len(w1) == 0 or len(w2) == 0:
        return 0
    for i in w1:
        for j in w2:
            score += i.path_similarity(j)
    return score/len(w1)/len(w2)

def Consinsimilarity(tensor1, tensor2):
    if tensor1.norm(dim = -1, keepdim = True) < 1e-3 or tensor2.norm(dim = -1, keepdim = True) < 1e-3:
        return 0
    normal_t1 = tensor1/tensor1.norm(dim = -1, keepdim = True)
    normal_t2 = tensor2/tensor2.norm(dim = -1, keepdim = True)
    return (normal_t1 * normal_t2).sum(dim = -1)

class Trees:
    def __init__(self, args, relations, rep):
        self.tree = None
        self.data = {}
        self.config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(args.bert_model, config = self.config)
        self.bert.eval()
        
        
        if args.dataset == "ace":
            from consts import LABEL_TRIGGERS_ACE, NUM_LABEL_ACE
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_ACE)
            self.num_label = NUM_LABEL_ACE
        elif args.dataset == "ere":
            from consts import LABEL_TRIGGERS_ERE, NUM_LABEL_ERE
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_ERE)
            self.num_label = NUM_LABEL_ERE
        elif args.dataset == "maven":
            from consts import LABEL_TRIGGERS_MAVEN
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_MAVEN)
        
        self.rep = rep
        self.tree = self.create(relations)

    def create(self, rel):
        linkmap = {}
        for item in rel:
            print(item)
            if item[1][0] not in linkmap.keys():
                linkmap[item[1][0]] = node(item[1][0].split(":")[-1], item[1][1])
                linkmap[item[1][0]].rep = self.rep[self.l_trigger2idx[item[1][0]]].detach()
            if item[0][0] not in linkmap.keys():
                linkmap[item[0][0]] = node(item[0][0].split(":")[-1],item[0][1])
            if linkmap[item[0][0]].rep == None:
                linkmap[item[0][0]].rep = linkmap[item[1][0]].rep
            else: 
                linkmap[item[0][0]].rep = torch.cat([linkmap[item[0][0]].rep, linkmap[item[1][0]].rep], dim = 0)
            linkmap[item[0][0]].child.append(linkmap[item[1][0]])
            linkmap[item[1][0]].parent = linkmap[item[0][0]]
        for i, n in linkmap.items():
            n.emb = torch.mean(n.rep, dim=0)
        
        for i, n in linkmap.items():
            height = 1
            n_b = n
            while n_b.parent != None:
                n.ancestors.append(n_b.parent)
                height += 1
                n_b = n_b.parent
            n.height = height
            assert n.height == len(n.ancestors) + 1
        with torch.no_grad():
            for i, v_n in linkmap.items():
                x = self.tokenizer.tokenize(v_n.name)
                v_n_id = self.tokenizer.convert_tokens_to_ids(x)
                segments_ids = torch.tensor([0] * len(v_n_id))
                v_n.name_vec = F.normalize(self.bert(torch.tensor([v_n_id]), token_type_ids = segments_ids)[1], dim=1).squeeze(dim=0).detach()
        for i,n in linkmap.items():
            if n.parent == None:
                return n
    
    def add_new(self, new_node):
        for item in new_node:
            for headnode in self.tree:
                self.search(item[1], headnode)
                self.temp_node.child.append(node(item[0],item[1]))



    def search_with_wordnet(self, emb, headnode, i, name):
        v = headnode.rep
        temp = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0)
        b = temp
        temp_node = None
        for children in headnode.child:
            v = children.rep
            r = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0)
            if r > temp:
                temp_node = children
                temp = r
        if temp > b + 1e-4:
            return self.search_with_wordnet(emb, temp_node, i, name)
        else:
            return headnode
        


    def search_pair(self, emb, headnode, i):
        v = headnode.rep
        temp = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0) / sqrt(headnode.height)
        b = temp
        temp_node = None
        for children in headnode.child:
            v = children.rep
            
            r = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0) / sqrt(children.height)
            
            if r > temp:
                temp_node = children
                temp = r
        if temp > b + 1e-4:
            return self.search_pair(emb, temp_node, i)
        else:
            return headnode




    def display(self,node):
        dict = {}
        dict["name"] = node.name
        dict["children"] = []
        for child in node.child:
            dict["children"].append(self.display(child))
        return dict




def link(args, structure, info_test, rep):
    t = Trees(args, structure, rep)
    res = {}
    for i, item in enumerate(info_test):
        
        if item["name"] == None:
            continue
        res[item["name"]] = {"sons":[],"fathers":[],"instance":[]}
        node = t.search_pair(item["vec"].detach(), t.tree, item["name"])
        while node.name != 'event_type':
            res[item["name"]]['fathers'].append(node.name)
            node = node.parent
        res[item["name"]]["instance"] = item["instance"]
    return res

from bert_score import score
def link_LLM(args, structure, info_test, rep):
    import openai
    import time
    openai.api_key = ""

    res = {}
    for i, item in enumerate(info_test):
        if item["name"] == None:
            continue
        
        res[item["name"]] = {"sons":[],"fathers":[],"instance":[]}
        if args.dataset == "ace":
            tree_node = [
            "Root",
            "Life",
            "Movement",
            "Transaction",
            "Conflict",
            "Contact",
            'Personnel',
            "Justice",
            'Justice:Trial-Hearing',
            'Life:Die',
            'Transaction:Transfer-Money',
            'Life:Injure',
            'Personnel:End-Position',
            'Personnel:Elect',
            'Contact:Meet',
            'Contact:Phone-Write',
            'Movement:Transport',
            'Conflict:Attack'
            ]
        elif args.dataset == "ere":
            tree_node = [
            'Root',
            'Conflict', 
            'Movement', 
            'Transaction', 
            'Life', 
            'Contact', 
            'Transaction', 
            'Personnel', 
            'Conflict:Attack', 
            'Movement:Transport-Person', 
            'Transaction:Transfer-Money', 
            'Contact:Contact', 
            'Life:Die', 
            'Contact:Broadcast', 
            'Transaction:Transfer-Ownership', 
            'Contact:Meet', 
            'Personnel:End-Position', 
            'Contact:Correspondence'
            ]
        elif args.dataset == "maven":
            tree_node = ['Violence', 'Attack', 'Military_operation', 'Hostile_encounter', 'Killing', 'Motion_vir', 'Motion', 'Self_motion', 'Arriving', 'Communication_vir', 'Statement', 'Action_vir', 'Social_event', 'Creating', 'Scenario', 'Catastrophe', 'Competition', 'Process_end', 'Process_start', 'Influence_vir', 'Causation', 'Conquering', 'AlterBadState', 'Bodily_harm', 'Destroying', 'Death', 'Change_vir', 'Coming_to_be', 'Root']
        
        tree = ",".join(tree_node)
        template =  """
        It is known that we have a new event type {} and a hierarchical Tree structure composed
        of these existing events [{}]. Please tell me which existing 
        events should be linked to if you want to add a new event type to this Tree structure? 
        Your answer should be one of these existing event types without any other word!Your answer should be one of these existing event types without any other word!Your answer should be one of these existing event types without any other word!
        , the following is an example:\n
        
        input word: Personnel:Nominate\n
        answer: Personnel\n

        input word: {}\n
        answer:
        """.format(item["name_LLM"][0], tree, item["name_LLM"][0])
        # the type of the trigger is a mask event
        ans = None
        time.sleep(20)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [
                {"role":"system","content":"you have the strong ability to name the event"},
                {"role":"user", "content":template}
            ],
            temperature = 0.2
        )
        ans = response["choices"][0]["message"]["content"].replace(" ","")
        
        if ans not in tree_node:
            ans = "Root"
        
        maxnode = ans
        
        if maxnode == "Root":
            res[item["name"]]['fathers'] = []
        else:
            if len(maxnode.split(":")) > 1:
                res[item["name"]]['fathers'] = [maxnode.split(":")[0], maxnode]
            else:
                res[item["name"]]['fathers'] = [maxnode]
        res[item["name"]]["instance"] = item["instance"]
    return res

def link_wordnet(args, structure, info_test, rep):
    t = Trees(args, structure, rep)
    res = {}
    for i, item in enumerate(info_test):
        if item["name"] == None:
            continue
        res[item["name"]] = {"sons":[],"fathers":[],"instance":[]}
        node = t.search_with_wordnet(item["vec"].detach(), t.tree, item["name"], item["name_LLM"][0])
        while node.name != 'event_type':
            res[item["name"]]['fathers'].append(node.name)
            node = node.parent
        res[item["name"]]["instance"] = item["instance"]
    return res

