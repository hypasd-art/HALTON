"""
Code modified from
https://github.com/wvangansbeke/Unsupervised-Classification
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature


    def mine_nearest_neighbors(self, topk, calculate_accuracy=False):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices,distances

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets,idx):
        for i,f in enumerate(features):
            self.features[idx[i]].copy_(f.detach())
        for i,t in enumerate(targets):
            self.targets[idx[i]].copy_(t.detach())

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, is_l):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        if not is_l :# unlabel
            idx = batch[-2]
            batch = batch[:-2]
        else: # label
            idx = batch[-1]
            batch = batch[:-1]
        label_ids = batch[3]
        feature = model.forward(batch, msg="feat")

        memory_bank.update(feature, label_ids, idx)
    print("finish filling memory bank")