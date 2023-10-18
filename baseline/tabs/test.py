import sys
path = "../../../HALTON/"
sys.path.append(path)

from tabs_dataloader import Labeled_Dataset as BertDataset
from tabs_dataloader import unLabeled_Dataset as PBertDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_tabs import *
from utils import seed_everything, data_split2, L2Reg, compute_kld,  _worker_init_fn_, build_vocab
import random
import os
from evaluation import ClusterEvaluation, usoon_eval, ACC
from sklearn.metrics.cluster import normalized_mutual_info_score
from memory import *
from utils import view_generator
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from consts import ace_structure_test, ace_structure, ere_structure, ere_structure_test, maven_structure_test, maven_structure
from tabs_link import *
from link import *
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, BertForMaskedLM
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import torch
import os
import numpy as np
from collections import Counter
    

# 先训练AUTOENCODER 10轮，再一起训
def load_pretrained(args):
    from transformers import BertTokenizer, BertModel, BertConfig
    config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model, config = config)
    generator = view_generator(tokenizer, args.rtr_prob, args.seed)
    return config, tokenizer, bert, generator

def test_one_epoch(net, args, epoch, new_class_dataloader):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        pseudos = []
        names = [[] for i in range(args.new_class)]
        res = [{"predict":[], "label":None} for i in range(args.new_class)]
        with tqdm(total=len(new_class_dataloader), desc='testing') as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                logits = net.module.encoder(data)
                name = net.module.get_name(data)
                # print(name)
                ground_truth.append(data[3])
                pred_label = logits.max(dim = -1)[1].cpu()
                label_pred.append(pred_label)

                for i in range(len(pred_label)):
                    names[pred_label[i]].extend(name[i].split())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy() 
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
            B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
            a = ACC(ground_truth, label_pred)
            print("acc:{}, B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}, NMI:{}".format(a, B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI,normalized_mutual_info_score(ground_truth, label_pred)))
            print(cluster_eval)
            # print(names)
            for i in range(len(names)):
                names[i] = Counter(names[i])
    # y_pred = torch.cat(label_predict, dim = 0).numpy()
    # y_true = torch.cat(ground_truth, dim = 0).numpy() 

    D = max(label_pred.max(), ground_truth.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(label_pred.size):
        w[label_pred[i], ground_truth[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    for i, j in zip(ind[0],ind[1]):
        res[i]["label"] = j

    for i in range(len(names)):
        for j in range(min(3, len(names[i]))):
            res[i]["predict"].append(names[i].most_common()[j][0])
            
    return cluster_eval['F1'], ARI, res

def name_cluster_generate(new_class_dataloader, info, t5_tokenizer, t5_mlm, link_res):
    t5_mlm.eval()
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        if len(item["using_ids"]) == 0:
            continue

        path = link_res[i]["fathers"]
        assert len(link_res[i]["instance"]) == len(item["ids"])
        path = ":".join(list(reversed(path)))

        idx = item["using_ids"][0]
        text = new_class_dataloader.dataset.examples[idx].text
        pos_span = new_class_dataloader.dataset.examples[idx].pos_span
        # assert item["label"] == new_class_dataloader.dataset.examples[idx].label
        trigger = text[pos_span[0]: pos_span[1]]
        text = " ".join(text)
        trigger = " ".join(trigger)
        template = text +  'According to this, the trigger word of this <extra_id_0> is '+ trigger +'.'
        # the type of the trigger is a mask event
        encoded = t5_tokenizer.encode_plus(template, add_special_tokens=True, truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].cuda()
        outputs = t5_mlm.generate(input_ids=input_ids, num_beams=200, num_return_sequences=5, max_length=5)

        
        end_token='<extra_id_1>'
        for output in outputs:
            _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if end_token in _txt:
                _end_token_index = _txt.index(end_token)
                res[i]["predict"].append(path+":"+_txt[:_end_token_index])
                res[i]["label"] = item["label"]
    print(res)
    return res

def name_cluster_generate_LMM(new_class_dataloader, info, link_res):
    import openai
    import time

    openai.api_key = "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"
    """
    export HTTPS_PROXY=http://cipzhao:cipzhao@210.75.240.136:10800
    export HTTP_PROXY=http://cipzhao:cipzhao@210.75.240.136:10800
    """
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        time.sleep(20)
        if len(item["using_ids"]) == 0:
            continue
        path = link_res[i]["fathers"]
        assert len(link_res[i]["instance"]) == len(item["ids"])
        path = ":".join(list(reversed(path)))
        idx = item["using_ids"][0]
        text = new_class_dataloader.dataset.examples[idx].text
        pos_span = new_class_dataloader.dataset.examples[idx].pos_span
        
        trigger = text[pos_span[0]: pos_span[1]]
        text = text[0:pos_span[0]] + text[pos_span[0]: pos_span[1]] + text[pos_span[1]:]
        text = " ".join(text)
        
        trigger = " ".join(trigger)
        template =  """
        you need to generate the event type according to the text and trigger. the event type you generate should be only one clear and brief word\n\
        some examples are as follows:\n\
        sentence: British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country's energy regulator as the new chairman of finance watchdog the Financial Services Authority (FSA).\n\
        the event trigger word is 'named'. According to this, the event name is what? \n\
        answer: Nominate\n
        sentence: As well as previously holding senior positions at Barclays Bank, BZW and Kleinwort Benson, McCarthy was formerly a top civil servant at the Department of Trade and Industry.\n\
        the event trigger word is 'previously'. According to this, the event name is what? \n\
        answer: End-Position\n
        sentence: The comments indicate that Russia's nuanced position over the war in Iraq was becoming ever more scrambled, with Putin pushing to protect his budding friendship with US President George W. Bush in the face of strident opposition from the Russian media and other top Kremlin officials.\n\
        the event trigger word is 'War'. According to this, the event name is what? \n\
        answer: Attack\n\
        then you need to generate the event type name to the following sentence:\n\
        """ + "sentence: " + text + "\nthe event trigger word is "+ trigger + '.' + 'According to this, the event name is what?\n answer:'
        # the type of the trigger is a mask event
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [
                {"role":"system","content":"you have the strong ability to name the event"},
                {"role":"user", "content":template}
            ],
            temperature = 0.2
        )
        res[i]["predict"].append(path+":"+response["choices"][0]["message"]["content"].lower())
        res[i]["label"] = item["label"]
        # print(res[i]["predict"])
    print(res)
    return res

def name_cluster_generate_LMM_with_path(new_class_dataloader, info, link_res):
    import openai
    import time

    openai.api_key = "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"
    """
    export HTTPS_PROXY=http://cipzhao:cipzhao@210.75.240.136:10800
    export HTTP_PROXY=http://cipzhao:cipzhao@210.75.240.136:10800
    """
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        time.sleep(20)
        if len(item["using_ids"]) == 0:
            continue
        path = link_res[i]["fathers"]
        assert len(link_res[i]["instance"]) == len(item["ids"])
        path = ":".join(list(reversed(path)))
        print(path)
        idx = item["using_ids"][0]
        text = new_class_dataloader.dataset.examples[idx].text
        pos_span = new_class_dataloader.dataset.examples[idx].pos_span
        
        trigger = text[pos_span[0]: pos_span[1]]
        text = text[0:pos_span[0]] + text[pos_span[0]: pos_span[1]] + text[pos_span[1]:]
        text = " ".join(text)
        
        trigger = " ".join(trigger)
        template =  """
        you need to generate the event type according to the text and trigger. the event type you generate should be only one clear and brief word\n\
        some examples are as follows:\n\
        sentence: British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country's energy regulator as the new chairman of finance watchdog the Financial Services Authority (FSA).\n\
        the event trigger word is 'named'. According to this, the event name is what? \n\
        answer: Personnel:Nominate\n
        sentence: As well as previously holding senior positions at Barclays Bank, BZW and Kleinwort Benson, McCarthy was formerly a top civil servant at the Department of Trade and Industry.\n\
        the event trigger word is 'previously'. According to this, the event name is what? \n\
        answer: Personnel:End-Position\n
        sentence: The comments indicate that Russia's nuanced position over the war in Iraq was becoming ever more scrambled, with Putin pushing to protect his budding friendship with US President George W. Bush in the face of strident opposition from the Russian media and other top Kremlin officials.\n\
        the event trigger word is 'War'. According to this, the event name is what? \n\
        answer: Confilct:Attack\n\
        then you need to generate the event type name to the following sentence:\n\
        """ + "sentence: " + text + "\nthe event trigger word is "+ trigger + '.' + 'According to this, the event name is what?\n answer:'+path+":"
        # the type of the trigger is a mask event
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [
                {"role":"system","content":"you have the strong ability to name the event"},
                {"role":"user", "content":template}
            ],
            temperature = 0.2
        )
        res[i]["predict"].append(path+":"+response["choices"][0]["message"]["content"].lower())
        res[i]["label"] = item["label"]
        # print(res[i]["predict"])
    print(res)
    return res


def name_cluster_generate_bert(new_class_dataloader, info, tokenizer, model, link_res):
    model.eval()
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        if len(item["using_ids"]) == 0:
            continue

        path = link_res[i]["fathers"]
        assert len(link_res[i]["instance"]) == len(item["ids"])
        path = ":".join(list(reversed(path)))
        
        idx = item["using_ids"][0]
        text = new_class_dataloader.dataset.examples[idx].text
        pos_span = new_class_dataloader.dataset.examples[idx].pos_span
        # assert item["label"] == new_class_dataloader.dataset.examples[idx].label
        trigger = text[pos_span[0]: pos_span[1]]
        text = " ".join(text)
        trigger = " ".join(trigger)

        template = text +  'According to this, the trigger word of this [MASK] is '+ trigger +'.'
        # the type of the trigger is a mask event
        tokenized_text = ['[CLS]'] + tokenizer.tokenize(template) + ['[SEP]']

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_tendor = torch.tensor([segments_ids]).cuda()
        masked_index = tokenized_text.index('[MASK]')

        # print(tokenized_text[masked_index])
        with torch.no_grad():
            predictions = model(tokens_tensor,token_type_ids = segments_tendor)
        # print(predictions)

        predcted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predict_token = tokenizer.convert_ids_to_tokens(predcted_index)

        res[i]["predict"].append(path+":"+predict_token)
        res[i]["label"] = item["label"]
    print(res)
    return res

def name_cluster_trigger(new_class_dataloader, info, link_res):
    import random
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        if len(item["ids"]) == 0:
            continue
        path = link_res[i]["fathers"]
        assert len(link_res[i]["instance"]) == len(item["ids"])
        path = ":".join(list(reversed(path)))
        # print(path)

        idx = item["ids"][random.randint(0,len(item['ids'])-1)]
        text = new_class_dataloader.dataset.examples[idx].text
        pos_span = new_class_dataloader.dataset.examples[idx].pos_span
        # assert item["label"] == new_class_dataloader.dataset.examples[idx].label
        trigger = text[pos_span[0]: pos_span[1]][0]
        res[i]["predict"].append(path+":"+trigger)
        res[i]["label"] = item["label"]
    print(res)
    return res

def name_cluster_trigger_majority(new_class_dataloader, info, link_res=None):
    res = [{"predict":[], "label":None} for i in range(len(info))]
    for i, item in enumerate(info):
        res[i]["label"] = item["label"]
        index = item["using_ids"]
        if len(index) == 0:
            continue
        
        if link_res!=None:
            path = link_res[i]["fathers"]
            assert len(link_res[i]["instance"]) == len(item["ids"])
            path = ":".join(list(reversed(path)))
        temp = []
        for idx in index:
            text = new_class_dataloader.dataset.examples[idx].text
            pos_span = new_class_dataloader.dataset.examples[idx].pos_span
            trigger = text[pos_span[0]: pos_span[1]][0]
            res[i]["predict"].append(trigger)
        result = Counter(res[i]["predict"])
        # label = Counter(label)
        if link_res!=None:
            res[i]["predict"]=[path+":"+result.most_common()[0][0]]
        else:
            res[i]["predict"]=[result.most_common()[0][0]]
    # print(res)
    return res


def get_top_n_instance(net, args, dataloader, n=5):
    net.eval()
    device=torch.device("cuda" if args.cuda else "cpu")

    with torch.no_grad():
        index = []
        label_predict = []
        ground_truth = []
        label = []
        true_label_count = [[] for i in range(args.new_class)]

        num = [0 for i in range(args.new_class)]
        centers = torch.zeros(args.new_class, 256, device=device)
        rep = [{"vec":[], "ids":[], "res":None, "using_ids":[], "label":None} for i in range(args.new_class)]
        for iteration, data in enumerate(dataloader):
            idx = data[-3]
            #data = data[:-2]
            true_label = data[3]
            logits = net.module.encoder(data)
            sia_rep, v2 = net.module.get_cluster_feat(data)
            sia_rep = sia_rep + v2
            label_pred = logits.max(dim = -1)[1].cpu()
            ground_truth.append(true_label)
            label_predict.append(label_pred)

            assert len(sia_rep) == len(label_pred)
            assert len(idx) == len(label_pred)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label_pred[i]
                centers[l] += vec
                true_label_count[l].append(true_label[i].item())
                rep[l]["vec"].append(vec)
                rep[l]["ids"].append(idx[i].item())
                num[l] += 1

        y_pred = torch.cat(label_predict, dim = 0).numpy()
        y_true = torch.cat(ground_truth, dim = 0).numpy() 

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(w.max() - w)
        for i, j in zip(ind[0],ind[1]):
            rep[i]["label"] = j

        for c in range(args.new_class):
            centers[c] /= num[c]
            true_label_count[c] = Counter(true_label_count[c])
            if len(rep[c]["vec"]) == 0:
                continue
            length_rep = len(rep[c]["vec"])
            rep[c]["vec"] = torch.stack(rep[c]["vec"], dim = 0)
            # print(rep[c]["vec"].shape)
            # print(centers[c].unsqueeze(0).shape)
            rep[c]["res"] = torch.mm(centers[c].unsqueeze(0), rep[c]["vec"].T).squeeze()
            rep[c]["vec"] = None
            # print(rep[c]["res"].shape)

            # assert len(rep[c]["ids"]) == rep[c]["res"].size(0)
        
            ans, argsort = torch.sort(rep[c]["res"], descending=True)
            argsort = argsort.cpu().numpy()
            if len(rep[c]["ids"]) == 1:
                rep[c]["using_ids"].append(rep[c]["ids"][argsort])
            else:
                for i in range(min(n, length_rep)):
                    rep[c]["using_ids"].append(rep[c]["ids"][argsort[i]])
            # rep[c]["label"] = true_label_count[c].most_common()[0][0]
    # print(rep)
    return rep, ind

def get_top_n_instance_gold(net, args, dataloader, n=5):
    net.eval()
    device=torch.device("cuda" if args.cuda else "cpu")

    with torch.no_grad():
        index = []
        label_predict = []
        ground_truth = []
        label = []
        true_label_count = [[] for i in range(args.new_class)]

        num = [0 for i in range(args.new_class)]
        centers = torch.zeros(args.new_class, 256, device=device)
        rep = [{"vec":[], "ids":[], "res":None, "using_ids":[], "label":None} for i in range(args.new_class)]
        for iteration, data in enumerate(dataloader):
            idx = data[-3]
            #data = data[:-2]
            true_label = data[3]
            logits = net.module.encoder(data)
            sia_rep, v2 = net.module.get_cluster_feat(data)
            sia_rep = sia_rep + v2
            label_pred = logits.max(dim = -1)[1].cpu()
            ground_truth.append(true_label)
            label_predict.append(label_pred)

            assert len(sia_rep) == len(label_pred)
            assert len(idx) == len(label_pred)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = true_label[i]
                centers[l] += vec
                true_label_count[l].append(true_label[i].item())
                rep[l]["vec"].append(vec)
                rep[l]["ids"].append(idx[i].item())
                num[l] += 1

        for c in range(args.new_class):
            centers[c] /= num[c]
            true_label_count[c] = Counter(true_label_count[c])
            if len(rep[c]["vec"]) == 0:
                continue
            length_rep = len(rep[c]["vec"])
            rep[c]["vec"] = torch.stack(rep[c]["vec"], dim = 0)
            # print(rep[c]["vec"].shape)
            # print(centers[c].unsqueeze(0).shape)
            rep[c]["res"] = torch.mm(centers[c].unsqueeze(0), rep[c]["vec"].T).squeeze()
            rep[c]["vec"] = None
            # print(rep[c]["res"].shape)

            # assert len(rep[c]["ids"]) == rep[c]["res"].size(0)
        
            ans, argsort = torch.sort(rep[c]["res"], descending=True)
            argsort = argsort.cpu().numpy()
            if len(rep[c]["ids"]) == 1:
                rep[c]["using_ids"].append(rep[c]["ids"][argsort])
            else:
                for i in range(min(n, length_rep)):
                    rep[c]["using_ids"].append(rep[c]["ids"][argsort[i]])
            # rep[c]["label"] = true_label_count[c].most_common()[0][0]
    # print(rep)
    return rep

 
from nltk.corpus import wordnet as wn
def calcu_path_sim(word1, word2):
    word1 = word1.split(":")[-1]
    word2 = word2.split(":")[-1]
    # print(word1)
    # print(word2)
    w1= wn.synsets(word1)
    w2 = wn.synsets(word2)
    score = 0
    if len(w1) == 0 or len(w2) == 0:
        return 0
    for i in w1:
        for j in w2:
            score += i.path_similarity(j)
    return score/len(w1)/len(w2)

from PyDictionary import PyDictionary
def get_word_definition(word):
    word1 = word1.split(" ")[0]
    word2 = word2.split(" ")[0]
    dictionary = PyDictionary()
    definition = dictionary.meaning(word)

    if definition:
        return definition
    else:
        return None

def compute_similary(query, refs, sent_sim_model):
    sent_sim_model.eval()
    embedding_query = sent_sim_model.encode(query, convert_to_tensor=True)
    similary = []
    for ref in refs:
        embedding_ref = sent_sim_model.encode(ref, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(embedding_query, embedding_ref)
        similary.append(torch.max(sim_matrix))
    return similary

def eval_bert_score(res, ref):
    P_single = 0
    R_single = 0
    F1_single = 0
    l = 0
    for rs in res:
        for rf in ref:
            for i in rs.split(":"):
                for j in rf.split(":"):
                    _P, _R, _F1 = score([i], [j], lang='en', rescale_with_baseline=True)
                    P_single += _P
                    R_single += _R
                    F1_single += _F1
                    l+=1
    return P_single/l, R_single/l, F1_single/l

def evaluate_name(args, res, dicts, naming_model, n = 3):
    l = len(res)
    P_single = 0
    R_single = 0
    F1_single = 0
    from consts import UNLABEL_TRIGGERS_ACE, UNLABEL_TRIGGERS_ERE, naming_unlabel_trigger_ace, naming_unlabel_trigger_ere, UNLABEL_TRIGGERS_MAVEN_NAME
    from rouge import Rouge
    if args.dataset == "ace":
        trigger_name_dict = UNLABEL_TRIGGERS_ACE
        trigger_dict = UNLABEL_TRIGGERS_ACE
    elif args.dataset == "ere":
        trigger_name_dict = UNLABEL_TRIGGERS_ERE
        trigger_dict = UNLABEL_TRIGGERS_ERE
    elif args.dataset == "maven":
        trigger_name_dict = UNLABEL_TRIGGERS_MAVEN_NAME
        trigger_dict = UNLABEL_TRIGGERS_MAVEN_NAME
    predict = []
    true = []
    def_pred = []
    def_true = []
    rouge = Rouge()
    rouge_f = 0
    path_similarity_score = 0
    for i in range(len(res)):
        if len(res[i]["predict"]) == 0:
            l-=1
            continue
        print(res[i]['predict'][0], trigger_name_dict[res[i]["label"]])
        # assert len(res[i]['predict']) == 1
        predict.append(res[i]['predict'][0])
        true.append(trigger_name_dict[res[i]["label"]].lower())
        rouge_res = rouge.get_scores(res[i]['predict'][0].lower().replace(":", " "), trigger_name_dict[res[i]["label"]].lower().replace(":", " "))
        # print(res[i]['predict'][0].replace(":", " "), trigger_name_dict[res[i]["label"]].lower().replace(":", " "))
        # print(rouge_res[0])
        rouge_f += rouge_res[0]['rouge-l']['f']
        # print(calcu_path_sim(res[i]['predict'][0], trigger_name_dict[res[i]["label"]].lower()))
        path_similarity_score += calcu_path_sim(res[i]['predict'][0], trigger_name_dict[res[i]["label"]].lower())
     
    path_similarity_score /= l
    rouge_f /= l
    _P, _R, _F1 = score(predict, true, lang='en', rescale_with_baseline=True)
    P_single += _P.mean()
    R_single += _R.mean()
    F1_single += _F1.mean()

    # P_single, R_single, F1_single = eval_bert_score(predict, true)
    print("rouge_f {}, single P R F1 {} {} {} path_sim {}".format(rouge_f, P_single,R_single,F1_single, path_similarity_score))


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
    net = torch.load(args.save+"model_tabs_"+ args.dataset +"_"+ str(args.seed) +"_not_norm_name.pt").cuda()
    epoch=0
    print("net ready...")
    print("-"*32)

    
    _,result,_ = test_one_epoch(net, args, epoch, new_class_train_dataloader)
    _,test_result,res_name = test_one_epoch(net, args, epoch, new_class_test_dataloader)
    print(res_name)
    rep = get_tree_info(net, args, known_class_train_dataloader)
    if args.use_graph:
        
        if args.dataset == "ace":
            info = create_info(args, ace_structure_test)
        elif args.dataset == "ere":
            info = create_info(args, ere_structure_test)
        elif args.dataset == "maven":
            info = create_info(args, maven_structure_test)
        gold_hierarchy_cluster_list = get_glod_hierarchy_cluster_list(info, new_class_test_dataloader, u_trigger2idx)

        info_new = get_test_info(args, new_class_test_dataloader, u_idx2trigger, net)
        info_gold = get_gold_info(args, new_class_test_dataloader, u_idx2trigger, net)
        assert len(info_new) == args.new_class
        info_name, ind = get_top_n_instance(net, args, new_class_test_dataloader)
        res_name = name_cluster_trigger_majority(new_class_test_dataloader, info_name)
        assert len(info_new) == len(res_name)
        for i,item in enumerate(info_new):
            assert len(item["instance"]) == len(info_name[i]['ids'])
            item["name_LLM"] = res_name[i]["predict"]
        info_name_gold = get_top_n_instance_gold(net, args, new_class_test_dataloader)
        res_name_gold = name_cluster_trigger_majority(new_class_test_dataloader, info_name_gold)
        for i,item in enumerate(info_gold):
            assert len(item["instance"]) == len(info_name_gold[i]['ids'])
            item["name_LLM"] = res_name_gold[i]["predict"]
        if args.dataset == "ace":
            # res = link(args, ace_structure, info_new, rep)
            # res_center = link_center(args, ace_structure, info_new, rep)
            res_wordnet = link_wordnet(args, ace_structure, info_new, rep)
            # res_with_name = link_with_name(args, ace_structure, info_new, rep)
            # res_gold = link(args, ace_structure, info_gold, rep)
            # res_gold_center = link_center(args, ace_structure, info_gold, rep)
            res_gold_wordnet = link_wordnet(args, ace_structure, info_gold, rep)
            # res_gold_with_name = link_with_name(args, ace_structure, info_gold, rep)
        elif args.dataset == "ere":
            # res = link(args, ere_structure, info_new, rep)
            # res_center = link_center(args, ere_structure, info_new, rep)
            res_wordnet = link_wordnet(args, ere_structure, info_new, rep)
            # res_with_name = link_with_name(args, ere_structure, info_new, rep)
            # res_gold = link(args, ere_structure, info_gold, rep)
            # res_gold_center = link_center(args, ere_structure, info_gold, rep)
            res_gold_wordnet = link_wordnet(args, ere_structure, info_gold, rep)
            # res_gold_with_name = link_with_name(args, ere_structure, info_gold, rep)
        elif args.dataset == "maven":
            # res = link(args, maven_structure, info_new, rep)
            # res_center = link_center(args, maven_structure, info_new, rep)
            res_wordnet = link_wordnet(args, maven_structure, info_new, rep)
            # res_with_name = link_with_name(args, maven_structure, info_new, rep)
            # res_gold = link(args, maven_structure, info_gold, rep)
            # res_gold_center = link_center(args, maven_structure, info_gold, rep)
            res_gold_wordnet = link_wordnet(args, maven_structure, info_gold, rep)
            # res_gold_with_name = link_with_name(args, maven_structure, info_gold, rep)
        test_data_num = len(new_class_test_dataloader.dataset)
        # predicted_cluster_list = get_predict_cluster_list(res)
        # predicted_cluster_list_center = get_predict_cluster_list(res_center)
        predicted_cluster_list_wordnet = get_predict_cluster_list(res_wordnet)
        # predicted_cluster_list_with_name = get_predict_cluster_list(res_with_name)
        # gold_cluster_list = get_predict_cluster_list(res_gold)
        # gold_cluster_list_center = get_predict_cluster_list(res_gold_center)
        gold_cluster_list_wordnet = get_predict_cluster_list(res_gold_wordnet)
        # gold_cluster_list_with_name = get_predict_cluster_list(res_gold_with_name)
# 
        # print("predicting result")
        # evaluation = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         predicted_cluster_list,
        #                                         test_data_num)
        # eval_info = evaluation.printEvaluation()
        # print("\n\n\n")
        # relation = evaluation.get_relation()
        # # print(relation)
        # print("predicting center result")
        # evaluation = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         predicted_cluster_list_center,
        #                                         test_data_num)
        # eval_info = evaluation.printEvaluation()
        # print("\n\n\n")

        print("predicting wordnet result")
        evaluation = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
                                                predicted_cluster_list_wordnet,
                                                test_data_num)
        eval_info = evaluation.printEvaluation()
        print("\n\n\n")

        # print("predicting name result")
        # evaluation = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         predicted_cluster_list_with_name,
        #                                         test_data_num)
        # eval_info = evaluation.printEvaluation()
        # print("\n\n\n")
        
        # print("gold result")
        # evaluation_gold = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         gold_cluster_list,
        #                                         test_data_num)
        # eval_info_gold = evaluation_gold.printEvaluation()
        # print("\n\n\n")

        # print("goldcenter result")
        # evaluation_gold = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         gold_cluster_list_center,
        #                                         test_data_num)
        # eval_info_gold = evaluation_gold.printEvaluation()
        # print("\n\n\n")

        print("gold wordnet result")
        evaluation_gold = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
                                                gold_cluster_list_wordnet,
                                                test_data_num)
        eval_info_gold = evaluation_gold.printEvaluation()
        print("\n\n\n")

        # print("gold name result")
        # evaluation_gold = HierarchyClusterEvaluation(gold_hierarchy_cluster_list,
        #                                         gold_cluster_list_with_name,
        #                                         test_data_num)
        # eval_info_gold = evaluation_gold.printEvaluation()
    
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_config = T5Config.from_pretrained('t5-base')
    naming_model = None #SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    t5_mlm = T5ForConditionalGeneration.from_pretrained('t5-base', config=t5_config).cuda()
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").cuda()
    # print(res)
    print("bert generate")
    res_name = name_cluster_generate_bert(new_class_test_dataloader, info_name, tokenizer, bert_model, res_wordnet)
    evaluate_name(args, res_name, u_idx2trigger, naming_model)

    print("T5 generate")
    res_name_T5 = name_cluster_generate(new_class_test_dataloader, info_name, t5_tokenizer, t5_mlm, res_wordnet)
    evaluate_name(args, res_name_T5, u_idx2trigger, naming_model)
    print("LLM generate with path")
    res_name_LMM_path = name_cluster_generate_LMM_with_path(new_class_test_dataloader, info_name, res_wordnet)
    evaluate_name(args, res_name_LMM_path, u_idx2trigger, naming_model)
    print("trigger")
    res_name_trg = name_cluster_trigger(new_class_test_dataloader, info_name, res_wordnet)
    evaluate_name(args, res_name_trg, u_idx2trigger, naming_model)

    print("trigger majority")
    res_name_tm = name_cluster_trigger_majority(new_class_test_dataloader, info_name, res_wordnet)
    evaluate_name(args, res_name_tm, u_idx2trigger, naming_model)

    
    
    # print("LLM generate")
    # res_name_LMM = name_cluster_generate_LMM(new_class_train_dataloader, info_name, res_wordnet)
    # evaluate_name(args, res_name_LMM, u_idx2trigger, naming_model)


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
    parser.add_argument("--b_size", type = int, default = 32)
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
    parser.add_argument("--use_graph", type=bool, default=True)
    parser.add_argument("--recon_loss", type = float, default = 0)
    parser.add_argument("--view_strategy",type = str,default = "none")
    parser.add_argument("--bert_model", 
                          default="bert-base-uncased", 
                          type=str,
                          help="bert pre-trained model selected in the list: bert-base-uncased, "
                          "bert-large-uncased, bert-base-cased, bert-large-cased")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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