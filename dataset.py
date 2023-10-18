from torch.utils.data import Dataset
from utils import clean_text
import torch
import json
import random
import numpy as np
import json

class InputExample(object):
  def __init__(self, unique_id, text, pos_span, label):
    self.unique_id = unique_id
    self.text = text # list
    self.pos_span = pos_span
    self.label = label
    self.pseudo = -1

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, pos_span, valid_mask, mask_span): 
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.pos_span = pos_span
    self.valid_mask = valid_mask
    self.mask_span = mask_span


class Labeled_Dataset(Dataset):
    def __init__(self,args, examples, tokenizer):
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)        
        print(len(self.examples))
        self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+actual_max_len, self.max_len)+30, tokenizer=self.tokenizer)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.features[index].input_ids, dtype = torch.long)
        input_mask = torch.tensor(self.features[index].input_mask, dtype = torch.long)
        valid_mask = torch.tensor(self.features[index].valid_mask, dtype = torch.long)
        label = self.examples[index].label
        pos_span = self.features[index].pos_span
        mask_span = self.features[index].mask_span
        return input_ids, input_mask, valid_mask, label, pos_span, mask_span, index

    def __len__(self):
        return len(self.examples)

    def get_neighbors(self,neighbor_list):
        input_ids = [] 
        input_mask = []
        valid_mask = []
        label = []
        pos_span = []
        mask_span = []
        index = []
        for i in neighbor_list:
            t_input_ids, t_input_mask, t_valid_mask, t_label, t_pos_span, t_mask_span, t_index = self.__getitem__(i)
            input_ids.append(t_input_ids)
            input_mask.append(t_input_mask)
            valid_mask.append(t_valid_mask)
            label.append(t_label)
            pos_span.append(t_pos_span)
            mask_span.append(t_mask_span)
            index.append(t_index)
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span)
        index = torch.LongTensor(index)
        return input_ids, input_mask,valid_mask, label, pos_span, mask_span, index

    def get_pos(self, pos_list):
        input_ids = [] 
        input_mask = []
        valid_mask = []
        label = []
        pos_span = []
        mask_span = []
        index = []
        for i in pos_list:
            while True:
                x = np.random.randint(0, len(self.examples))
                t_input_ids, t_input_mask, t_valid_mask, t_label, t_pos_span, t_mask_span, t_index = self.__getitem__(x)
                if t_label == i:
                    break
            input_ids.append(t_input_ids)
            input_mask.append(t_input_mask)
            valid_mask.append(t_valid_mask)
            label.append(t_label)
            pos_span.append(t_pos_span)
            mask_span.append(t_mask_span)
            index.append(t_index)
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span) # (batch_size, 2)
        index = torch.LongTensor(index)
        return input_ids, input_mask,valid_mask, label, pos_span, mask_span, index

    def get_neg(self, neg_list):
        input_ids = [] 
        input_mask = []
        valid_mask = []
        label = []
        pos_span = []
        mask_span = []
        index = []
        for i in neg_list: 
            while True:
                x = np.random.randint(0, len(self.examples))
                t_input_ids, t_input_mask, t_valid_mask, t_label, t_pos_span, t_mask_span, t_index = self.__getitem__(x)
                if t_label != i:
                    break
            input_ids.append(t_input_ids)
            input_mask.append(t_input_mask)
            valid_mask.append(t_valid_mask)
            label.append(t_label)
            pos_span.append(t_pos_span)
            mask_span.append(t_mask_span)
            index.append(t_index)
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span) # (batch_size, 2)
        index = torch.LongTensor(index)
        return input_ids, input_mask, valid_mask, label, pos_span, mask_span, index

    def collate_fn(data):
        data = list(zip(*data))
        input_ids, input_mask, valid_mask, label, pos_span, mask_span, index = data
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span) # (batch_size, 2)
        index = torch.LongTensor(index)
        return input_ids, input_mask,valid_mask, label, pos_span, mask_span, index

    def preprocess(path, dicts, event_dict):
        datas = []
        with open(path,'r') as f:
            data = json.load(f)
            unique_id = 0
            for item in data:
                for event_mention in item['golden-event-mentions']:
                    if event_mention["event_type"] in dicts:
                        text = item['words']
                        pos_span = [event_mention["trigger"]["start"],event_mention["trigger"]["end"]]
                        event_type = event_mention["event_type"]
                        datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                        unique_id += 1
        return datas

    def preprocess_ere(path,dict,event_dict):
        datas = []
        unique_id = 0
        with open(path,"r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item['event_mentions']:
                    if event_mention["event_type"] in dict:
                        text = item['tokens']
                        pos_span = [event_mention["trigger"]["start"],event_mention["trigger"]["end"]]
                        event_type = event_mention["event_type"]
                        datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                        unique_id += 1 
        return datas
    
    
    def preprocess_maven(path,dict,event_dict):
        datas = []
        unique_id = 0
        num_samples = [0] * len(event_dict)
        with open(path,"r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item['events']:
                    if event_mention["type"] in dict:
                        for i in range(len(event_mention["mention"])):
                            text = item['content'][int(event_mention["mention"][i]["sent_id"])]["tokens"]
                            pos_span = [event_mention["mention"][i]["offset"][0],event_mention["mention"][i]["offset"][1]]
                            event_type = event_mention["type"]
                            num_samples[event_dict[event_type]] += 1
                            if num_samples[event_dict[event_type]] > 300:
                                continue
                            else:
                                datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                                unique_id += 1 
        return datas
    
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len-2:
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len
    
    def convert_examples_to_features(self, examples, seq_length, tokenizer, prompt_type=3):
        features = []
        for example in examples:
            tokens = []
            valid_mask = [0]
            tokens.append("[CLS]")

            trigger_tokens = example.text[example.pos_span[0]: example.pos_span[1]]
            if prompt_type == 1:
                prompt = trigger_tokens + ['is','a', tokenizer.mask_token, 'event']  
                mask_word_prefix = example.text + trigger_tokens  + ['is', 'a']
            elif prompt_type == 2:
                prompt = [ 'According', 'to', 'this', ',', 'the', 'trigger','word', 'of', 'this', tokenizer.mask_token, 'is'] +trigger_tokens +['.']
                mask_word_prefix = example.text + [ 'According', 'to', 'this', ',', 'the', 'trigger','word', 'of', 'this']
                # 'According to this, the trigger word of this mask is '+ trigger +'.'
            elif prompt_type == 3:
                prompt = ["the", "type", "of", "the"] + trigger_tokens + ["is", "a", tokenizer.mask_token, "event"]
                mask_word_prefix = example.text + ["the", "type", "of", "the"] + trigger_tokens + ["is", "a"]
            all_tokens = example.text + prompt
            
            for word in all_tokens:
                word = tokenizer.tokenize(word)
                tokens.extend(word)
                valid_mask.extend([1]+[0]*(len(word)-1))
            if len(tokens) > seq_length - 1:
                tokens = tokens[0 : (seq_length - 1)]
            valid_mask.extend([0])
            tokens.append("[SEP]")
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # print(tokens)
            token_ids = tokenizer.encode(' '.join(all_tokens), return_tensors='pt').squeeze(0)

            
            assert token_ids.size(0) == len(input_ids)
            prefix_bpe = tokenizer.encode(' '.join(mask_word_prefix))
            mask_span = len(prefix_bpe) -1

            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
            while len(valid_mask) < seq_length:
                valid_mask.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            features.append(
                InputFeatures(
                    unique_id = example.unique_id,
                    tokens = tokens, # bert_token
                    input_ids = input_ids,
                    input_mask = input_mask,
                    pos_span = example.pos_span,
                    valid_mask = valid_mask,
                    mask_span = mask_span
                    )
                )
        return features


class unLabeled_Dataset(Dataset):
    def __init__(self, args, examples, tokenizer):
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)        
        print(len(self.examples))
        self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+actual_max_len, self.max_len)+30, tokenizer=self.tokenizer)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.features[index].input_ids, dtype = torch.long)
        input_mask = torch.tensor(self.features[index].input_mask, dtype = torch.long)
        valid_mask = torch.tensor(self.features[index].valid_mask,dtype = torch.long)
        label = self.examples[index].label
        pos_span = self.features[index].pos_span
        pseudo = self.examples[index].pseudo
        mask_span = self.features[index].mask_span
        return input_ids, input_mask, valid_mask, label, pos_span, mask_span, index, pseudo

    def __len__(self):
        return len(self.examples)
    
    def get_neighbors(self,neighbor_list):
        input_ids = [] 
        input_mask = []
        valid_mask = []
        label = []
        pos_span = []
        mask_span = []
        index = []
        pseudo = []
        for i in neighbor_list:
            t_input_ids, t_input_mask, t_valid_mask, t_label, t_pos_span, t_mask_span, t_index, t_pseudo = self.__getitem__(i)
            input_ids.append(t_input_ids)
            input_mask.append(t_input_mask)
            valid_mask.append(t_valid_mask)
            label.append(t_label)
            pos_span.append(t_pos_span)
            mask_span.append(t_mask_span)
            index.append(t_index)
            pseudo.append(t_pseudo)
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span) # (batch_size, 2)
        index = torch.LongTensor(index)
        pseudo = torch.LongTensor(pseudo)
        return input_ids, input_mask,valid_mask, label, pos_span, mask_span, index, pseudo

    def collate_fn(data):
        data = list(zip(*data))
        input_ids, input_mask, valid_mask, label, pos_span,mask_span, index, pseudo = data
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        valid_mask = torch.stack(valid_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        pos_span = torch.LongTensor(pos_span) # (batch_size, 2)
        mask_span = torch.LongTensor(mask_span) # (batch_size, )
        index = torch.LongTensor(index)
        pseudo = torch.LongTensor(pseudo)
        return input_ids, input_mask, valid_mask, label, pos_span, mask_span, index, pseudo

    def preprocess(path,dict,event_dict):
        datas = []
        with open(path,'r') as f:
            data = json.load(f)
            unique_id = 0
            for item in data:
                for event_mention in item['golden-event-mentions']:
                    if event_mention["event_type"] in dict:
                        text = item['words']
                        pos_span = [event_mention["trigger"]["start"],event_mention["trigger"]["end"]]
                        event_type = event_mention["event_type"]
                        datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                        unique_id += 1
        return datas

    def preprocess_ere(path,dict,event_dict):
        datas = []
        unique_id = 0
        with open(path,"r") as f:
            for line in f.readlines():

                item = json.loads(line)
                for event_mention in item['event_mentions']:
                    if event_mention["event_type"] in dict:
                        text = item['tokens']
                        pos_span = [event_mention["trigger"]["start"],event_mention["trigger"]["end"]]
                        event_type = event_mention["event_type"]
                        datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                        unique_id += 1 
        return datas

    
    def preprocess_maven(path,dict,event_dict):
        datas = []
        unique_id = 0
        num_samples = [0] * len(event_dict)
        with open(path,"r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item['events']:
                    if event_mention["type"] in dict:
                        for i in range(len(event_mention["mention"])):
                            text = item['content'][int(event_mention["mention"][i]["sent_id"])]["tokens"]
                            pos_span = [event_mention["mention"][i]["offset"][0],event_mention["mention"][i]["offset"][1]]
                            event_type = event_mention["type"]
                            num_samples[event_dict[event_type]] += 1
                            if num_samples[event_dict[event_type]] > 300:
                                continue
                            else:
                                datas.append(InputExample(unique_id, text, pos_span, event_dict[event_type]))
                                unique_id += 1 
        return datas
    
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len-2:
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len
    
    def convert_examples_to_features(self, examples, seq_length, tokenizer, prompt_type=3):
        features = []
        for example in examples:
            tokens = []
            valid_mask = [0]
            tokens.append("[CLS]")

            trigger_tokens = example.text[example.pos_span[0]: example.pos_span[1]]
            if prompt_type == 1:
                prompt = trigger_tokens + ['is','a', tokenizer.mask_token, 'event']  
                mask_word_prefix = example.text + trigger_tokens  + ['is', 'a']
            elif prompt_type == 2:
                prompt = [ 'According', 'to', 'this', ',', 'the', 'trigger','word', 'of', 'this', tokenizer.mask_token, 'is'] +trigger_tokens +['.']
                mask_word_prefix = example.text + [ 'According', 'to', 'this', ',', 'the', 'trigger','word', 'of', 'this']
            elif prompt_type == 3:
                prompt = ["the", "type", "of", "the"] + trigger_tokens + ["is", "a", tokenizer.mask_token, "event"]
                mask_word_prefix = example.text + ["the", "type", "of", "the"] + trigger_tokens + ["is", "a"]
            all_tokens = example.text + prompt
            
            for word in all_tokens:
                word = tokenizer.tokenize(word)
                tokens.extend(word)
                valid_mask.extend([1]+[0]*(len(word)-1))
            if len(tokens) > seq_length - 1:
                tokens = tokens[0 : (seq_length - 1)]
            valid_mask.extend([0])
            tokens.append("[SEP]")
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            
            token_ids = tokenizer.encode(' '.join(all_tokens), return_tensors='pt').squeeze(0)
            assert token_ids.size(0) == len(input_ids)
            prefix_bpe = tokenizer.encode(' '.join(mask_word_prefix))
            mask_span = len(prefix_bpe) -1

            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
            while len(valid_mask) < seq_length:
                valid_mask.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            features.append(
                InputFeatures(
                    unique_id = example.unique_id,
                    tokens = tokens, # bert_token
                    input_ids = input_ids,
                    input_mask = input_mask,
                    pos_span = example.pos_span,
                    valid_mask = valid_mask,
                    mask_span = mask_span
                    )
                )
        return features
