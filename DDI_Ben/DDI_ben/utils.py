import os
import numpy as np
from models.CompGCN import *
from models.SkipGNN import *
from models.KGE import *
from models.MLP import *
from models.KGDDI import *
from models.CSMDDI import *
from models.HINDDI import *
from models.Decagon import *
from models.TIGER.randomWalk import *
from models.TIGER.model.tiger import *
import json
import sys
import torch
import time

import random

import fcntl
import pandas
import pickle as pkl

from collections import defaultdict as ddict

from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import torch.optim as optim

from rdkit import Chem
import networkx as nx
from torch_geometric.utils import subgraph, degree, get_laplacian

# from randomWalk import Node2vec

ddi_statistic = {
    'Pharmacokinetic interactions:Absorption interactions': [2, 12, 17, 61, 66],
    'Pharmacokinetic interactions:Distribution interactions': [42, 44, 72, 74],
    'Pharmacokinetic interactions:Metabolic interactions': [3, 10, 46],
    'Pharmacokinetic interactions:Excretion interactions':[64, 71],
    'Pharmacodynamic interactions:Additive or synergistic effects':[0, 1, 5, 6, 7, 8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 45, 51, 52, 53, 54, 55, 56, 58, 59, 62, 63, 67, 68, 70, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
    'Pharmacodynamic interactions:Antagonistic effects':[4, 11, 13, 16, 25, 28, 36, 47, 48, 49, 50, 57, 60, 65, 69, 75]
}

def load_data(args):
    if 'drugbank' in args.dataset:
        if args.dataset_type == 'finger':
            if args.adversarial and args.adversarial_mode == 2:
                folder_name = args.dataset + '_' + args.dataset_type + '_' + str(args.gamma_split) + '_2'
            else:
                folder_name = args.dataset + '_' + args.dataset_type + '_' + str(args.gamma_split)
        elif args.dataset_type == 'random':
            if args.adversarial and args.adversarial_mode == 2:
                folder_name = args.dataset + '_' + args.dataset_type + '_2'
            else:
                folder_name = args.dataset + '_' + args.dataset_type
        else:
            folder_name = args.dataset + '_' + args.dataset_type
    else:
        if args.dataset_type == 'finger':
            if args.adversarial and args.adversarial_mode == 2:
                folder_name = args.dataset + '_' + args.dataset_type + '_' + str(args.gamma_split) + '_2'
            elif args.adversarial and args.adversarial_mode == 1:
                folder_name = args.dataset + '_' + args.dataset_type + '_' + str(args.gamma_split) + '_1'
            else:
                folder_name = args.dataset + '_' + args.dataset_type + '_' + str(args.gamma_split)
        elif args.dataset_type == 'random':
            if args.adversarial and args.adversarial_mode == 2:
                folder_name = args.dataset + '_' + args.dataset_type + '_2'
            elif args.adversarial and args.adversarial_mode == 1:
                folder_name = args.dataset + '_' + args.dataset_type + '_1'
            else:
                folder_name = args.dataset + '_' + args.dataset_type
        else:
            folder_name = args.dataset + '_' + args.dataset_type
    if 'drugbank' in args.dataset:
        if args.DDIsetting == 'S0': ### not use now
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[]}
            sets = ['train', 'valid_S0', 'test_S0']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        elif args.DDIsetting == 'S1': ### not use now
            triple_dict = {'train':[], 'valid_S1':[], 'test_S1':[]}
            sets = ['train', 'valid_S1', 'test_S1']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        elif args.DDIsetting == 'S2': ### not use now
            triple_dict = {'train':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        else:
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[], 'valid_S1':[], 'test_S1':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
            if args.adversarial and args.adversarial_mode == 1:
                triple_dict['adv_1'] = []
                sets.append('adv_1')
            if args.adversarial and args.adversarial_mode == 2:
                triple_dict['adv_2'] = []
                sets.append('adv_2')
            for i in sets:
                if i == 'train' and args.adversarial and args.adversarial_mode == 1:
                    file = open('./data/{}/{}.txt'.format(folder_name, i + '_1'))
                else:
                    file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
    elif 'twosides' in args.dataset:
        if args.DDIsetting == 'S0': ### not use now
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[]}
            sets = ['train', 'valid_S0', 'test_S0']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        elif args.DDIsetting == 'S1': ### not use now
            triple_dict = {'train':[], 'valid_S1':[], 'test_S1':[]}
            sets = ['train', 'valid_S1', 'test_S1']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        elif args.DDIsetting == 'S2': ### not use now
            triple_dict = {'train':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        else:
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[], 'valid_S1':[], 'test_S1':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
            if args.adversarial and args.adversarial_mode == 1:
                triple_dict['adv_1'] = []
                sets.append('adv_1')
            if args.adversarial and args.adversarial_mode == 2:
                triple_dict['adv_2'] = []
                sets.append('adv_2')
            for i in sets:
                file = open('./data/{}/{}.txt'.format(folder_name, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
    return triple_dict

def rel_to_ddi_class():
    ddi_to_class = {j: i for i in ddi_statistic for j in ddi_statistic[i]}
    class_type = [j for j in ddi_statistic]
    return ddi_to_class, class_type

# def load_data(args):
#     if 'drugbank' in args.dataset:
#         triple_dict = {'train':[], 'valid':[], 'test':[]}
#         sets = ['train', 'valid', 'test']
#         for i in sets:
#             if i == 'train':
#                 file_name = i
#             else:
#                 file_name = i + '_' + args.DDIsetting
#             file = open('./data/{}/{}.txt'.format(args.dataset, file_name))
#             for j in file:
#                 str_lin = j.strip().split(' ')
#                 triple_dict[i].append([int(j) for j in str_lin])
#             file.close()
#     return triple_dict

def load_feature(args):
    feat = 0
    if 'drugbank' in args.dataset:
        with open('data/initial/drugbank/DB_molecular_feats.pkl', 'rb') as f:
            x = pkl.load(f, encoding='utf-8')
        # node_feat = torch.FloatTensor(x)
        feat = []
        for y in x['Morgan_Features']:
            feat.append(y)
    if 'twosides' in args.dataset:
        with open('data/initial/twosides/DB_molecular_feats.pkl', 'rb') as f:
            feat = pkl.load(f, encoding='utf-8')
    return feat

def add_model(args, data_record, device):
    model = 0
    if args.model == 'CompGCN':
        if args.Comp_sfunc == 'TransE':
            model = CompGCN_TransE_DDI(data_record.edge_index, data_record.edge_type, args, data_record.feat)
        model.to(device)
    elif args.model == 'SkipGNN':
        model = SkipGNN(data_record.feat.shape[1], args.skip_hidden, args.skip_dropout, data_record.num_rel, args).to(device)
    elif args.model in ['ComplEx', 'MSTE']:
        if args.model == 'ComplEx':
            model = KGEModel('ComplEx', data_record.num_ent, data_record.num_rel, args).to(device)
        elif args.model == 'MSTE':
            model = KGEModel('MSTE', data_record.num_ent, data_record.num_rel, args).to(device)
    elif args.model == 'MLP':
        model = MLP(data_record.num_ent, data_record.num_rel, args.mlp_dim, args, data_record.feat).to(device)
    elif args.model == 'KGDDI':
        model = KGDDI_MLP(data_record.num_ent, data_record.num_rel, args.kgddi_dim, args, data_record.feat).to(device)
    elif args.model == 'CSMDDI':
        model = CSMDDI(data_record.num_ent, data_record.num_rel, args.csm_dim, args, data_record, device)
    elif args.model == 'HINDDI':
        model = HIN_MLP(data_record.num_ent, data_record.num_rel, args.hin_dim, args, data_record.meta_feature).to(device)
    elif args.model == 'Decagon':
        model = Decagon(data_record.edge_index, data_record.num_rel, args.decagon_dim, data_record.feat, args).to(device)
    elif args.model == 'TIGER':
        # model = TIGER(max_layer=args.layer,
        #               num_features_drug = 67,
        #               num_nodes=dataset_statistics['num_nodes'],
        #               num_relations_mol=dataset_statistics['num_rel_mol'],
        #               num_relations_graph=dataset_statistics['num_rel_graph'],
        #               output_dim=args.d_dim,
        #               max_degree_graph=dataset_statistics['max_degree_graph'],
        #               max_degree_node = dataset_statistics['max_degree_node'],
        #               sub_coeff=args.sub_coeff,
        #               mi_coeff=args.mi_coeff,
        #               dropout=args.dropout,
        #               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = TIGER(max_layer=2,
                      num_features_drug = 67,
                      num_nodes = data_record.TG_data_sta['num_nodes'],
                      num_relations_mol = data_record.TG_data_sta['num_rel_mol'],
                      num_relations_graph = data_record.TG_data_sta['num_rel_graph'],
                      output_dim=64,
                      max_degree_graph = data_record.TG_data_sta['max_degree_graph'],
                      max_degree_node = data_record.TG_data_sta['max_degree_node'],
                      sub_coeff=0.1,
                      mi_coeff=0.1,
                      dropout=0.2,
                      device=device,
                      num_rel = data_record.num_rel,
                      args = args)
        model.to(device)
    return model

def read_batch(batch, split, device, args, data_record = None):
    # if args.model == 'CompGCN':
    if args.model in ['CompGCN', 'MLP', 'CSMDDI', 'HINDDI', 'Decagon']:
        if split == 'train':
            triple, label = [ _.to(device) for _ in batch]
            return [triple[:, 0], triple[:, 1], triple[:, 2]], label
        else:
            triple, label = [ _.to(device) for _ in batch]
            return [triple[:, 0], triple[:, 1], triple[:, 2]], label
    elif args.model == 'SkipGNN':
        # if split == 'train':
        triple, label = [ _.to(device) for _ in batch]
        return [data_record.feat.to(device), data_record.adj.to(device), data_record.adj2.to(device), [triple[:, 0], triple[:, 1]]], label
    elif args.model in ['ComplEx', 'MSTE']:
        triple, label = [ _.to(device) for _ in batch]
        if args.dataset == 'drugbank':
            num_rel = data_record.num_rel
            neg_data = []
            samp_set_0 = [i for i in range(num_rel)]
            for j in triple:
                samp_set = list(set(samp_set_0) - set([j[2].item()]))
                # samp_set = [i for i in range(num_rel) if i != j[2].item()]
                n_neg = 1 if args.model == 'MSTE' else 16
                # n_neg = 16
                neg_data.append(random.sample(samp_set, n_neg))
            neg_data = torch.LongTensor(neg_data).to(device)
            return [triple, neg_data, split], label
        elif args.dataset == 'twosides':
            if split == 'train':
                return [triple[:,:3], triple[:,3:], split], label
            else:
                return [triple, 0, split], label
    elif args.model in ['KGDDI']:
        if args.KGDDI_pre == 1:
            triple, label = [ _.to(device) for _ in batch]
            num_ent = args.num_bnent
            neg_data = []
            samp_set_0 = [i for i in range(num_ent)]
            for j in triple:
                samp_set = list(set(samp_set_0) - set([j[2].item()]))
                n_neg = 16
                neg_data.append(random.sample(samp_set, n_neg))
            neg_data = torch.LongTensor(neg_data).to(device)
            return [triple, neg_data, split], label
        else:
            if split == 'train':
                triple, label = [ _.to(device) for _ in batch]
                return [triple[:, 0], triple[:, 1], triple[:, 2]], label
            else:
                triple, label = [ _.to(device) for _ in batch]
                return [triple[:, 0], triple[:, 1], triple[:, 2]], label
    elif args.model in ['TIGER']:
        # return batch, get_label_ddi(batch[0].y,data_record.num_rel).to(device)
        if args.dataset == 'drugbank':
            return batch[:4], get_label_ddi(batch[4],data_record.num_rel).to(device)
        elif args.dataset == 'twosides':
            return batch[:4], batch[4].to(device)

### for SkipGNN
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def savePickleReader(file):
    if os.path.exists(file):
        while True:
            try:
                with open(file, "rb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    unpickler = pkl.Unpickler(f)
                    data = unpickler.load()
                    f.close()
                    break
            except:
                continue
        return data
    else:
        return None

### for KG-DDI
def pretrain_bn(args, device, file_name):
    ### load and set data
    if args.bionet == 'HetioNet':
        BN_path = 'relations_2hop'
    elif args.bionet == 'PrimeKG':
        BN_path = 'BN_Primekg'
    triple_bn = []
    file = open('./data/{}/{}.txt'.format(args.dataset, BN_path)) ###relations_2hop/relations_2hop_small
    for j in file:
        str_lin = j.strip().split(' ')
        triple_bn.append([int(j) for j in str_lin])
    triple_bn_np = np.array(triple_bn)[:,[0,2,1]] ### swap the second and third column
    num_bn = len(triple_bn)
    list_all = [j for j in range(num_bn)]
    train_num = int(num_bn / 10 * 9.9)
    train_set = random.sample(list_all, int(train_num))
    list_all = list(set(list_all) - set(train_set))
    valid_set = list_all

    num_bnent = triple_bn_np.max() + 1
    num_bnrel = triple_bn_np[:, 1].max() + 1

    train_triplets = triple_bn_np[train_set]
    valid_triplets = triple_bn_np[valid_set]

    data = ddict(list)
    sr2o = ddict(set)

    for j in train_triplets:
        sub, rel, obj = j[0], j[1], j[2]
        data['train'].append((sub, rel, obj))
        sr2o[(sub, rel)].add(obj)
        # data['train'].append((obj, rel + num_bnrel, sub))
        # sr2o[(obj, rel + num_bnrel)].add(sub)
    
    sr2o_train = {k: list(v) for k, v in sr2o.items()}

    for j in valid_triplets:
        sub, rel, obj = j[0], j[1], j[2]
        data['valid'].append((sub, rel, obj))
        sr2o[(sub, rel)].add(obj)
        # data['valid'].append((obj, rel + num_bnrel, sub))
        # sr2o[(obj, rel + num_bnrel)].add(sub)

    sr2o_all = {k: list(v) for k, v in sr2o.items()}

    triples  = ddict(list)

    for (sub, rel), obj in sr2o_train.items():
        triples['train'].append({'triple':(sub, rel, -1), 'label': sr2o_train[(sub, rel)], 'sub_samp': 1})

    for sub, obj, rel in data['valid']:
        triples['valid'].append({'triple': (sub, obj, rel), 	   'label': sr2o_all[(sub, obj)]})
    
    triples = dict(triples)

    data_iter_train = get_data_loader(triples, args, KGDDI_TrainDataset, 'train', 512)
    data_iter_valid = get_data_loader(triples, args, KGDDI_TestDataset, 'valid', 512)

    ### begin the training step
    ### model name: 'TransE', 'ComplEx'
    pre_model = KGDDI_pretrain('ComplEx', num_bnent, num_bnrel, args).to(device)
    pre_optimizer = optim.AdamW(pre_model.parameters(), lr=0.0001, weight_decay=args.weight_decay)

    # self.model = add_model(args, self.data_record, self.device) ###

    # self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay) ###

    best_val_mrr = 0
    mrr_change = 0

    args.num_bnent = num_bnent
    args.KGDDI_pre = 1

    for epoch in range(20):
        train_iter = iter(data_iter_train)
        pre_model.train()
        loss_list = []
        for step, batch in enumerate(train_iter):
            data, label = read_batch(batch, 'train', device, args)
            loss = pre_model.train_step(pre_model, pre_optimizer, data)
            loss_list.append(loss.item())
        print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, Loss: {}'.format(epoch, np.mean(np.array(loss_list))))
        with open(os.path.join('record', file_name), 'a+') as f:
            f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, Loss: {}'.format(epoch, np.mean(np.array(loss_list))))
        pre_model.eval()
        valid_iter = iter(data_iter_valid)
        mrr_list = []
        for step, batch in enumerate(valid_iter):
            data, label = read_batch(batch, 'valid', device, args)
            mrr = pre_model.test_step(pre_model, data, label)
            mrr_list.append(mrr)
        mrr_final = np.concatenate(mrr_list).mean()
        print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, MRR: {}'.format(epoch, mrr_final))
        with open(os.path.join('record', file_name), 'a+') as f:
            f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, MRR: {}'.format(epoch, mrr_final))
        if mrr_final > best_val_mrr:
            best_val_mrr = mrr_final
            pre_model.update_emb()
        #     mrr_change = 0
        # else:
        #     mrr_change += 1
        #     if mrr_change == 20:
        #         break
    emb_final = pre_model.return_emb()
    return emb_final

def get_data_loader(triples, args, dataset_class, split, batch_size, shuffle=True):
    return  DataLoader(
        dataset_class(triples[split], args),
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = 10, ### set the default numworkers to 10
        collate_fn      = dataset_class.collate_fn
    )

class KGDDI_TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		# self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		label, sub_samp	= np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		trp_label		= self.get_label_ddi(label)
		triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_label_ddi(self, label):
		# y = np.zeros([self.p.num_rel * 2], dtype=np.float32)
		y = np.zeros([self.p.num_bnent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class KGDDI_TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		label		= self.get_label_ddi(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_bnent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

### for CSMDDI
def train_wo_batch(model, optimizer, device, args, data_record):
    loss = model.forward_to()
    loss.backward()
    optimizer.step()
    return loss

### for TIGER

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(datapath, ligands):

    smile_graph = {}

    paths = datapath + "/tiger_mol_sp.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            if smile_graph[s] == [0, 0, 0, 0, 0, 0, 0, 0]: continue
            max_rel = max(smile_graph[s][6]) if max(smile_graph[s][6]) > max_rel else max_rel
            max_degree = smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree

        return smile_graph, max_rel, max_degree

    smiles_max_node_degree = []
    num_rel_mol_update = 0
    for d in ligands.keys():
        # if d == '997':
        #     xxx = 0
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))  ##还是smiles序列
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(lg)
        # if c_size == 0: ##证明这个药物只由一个atom组成，这种的不考虑
        #     continue
        if s_value == 0:
            pass
        elif max(s_value) > num_rel_mol_update:
            num_rel_mol_update = max(s_value)
        smile_graph[d] = c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg
        smiles_max_node_degree.append(deg)

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, num_rel_mol_update, max(smiles_max_node_degree)

def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2]
    mol_rel_index = mol_index[:,2]

    ##在这个位置应该计算的是最短路径
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value
    s_rel[np.where(s_value == 1)] = mol_rel_index  ##将直接相连的关
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    ##c_size:原子的个数
    ##features:每个原子的特征 c_size * 67
    ##edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist(), s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(), max(degrees)

def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)

def read_network(path):

    edge_index = []
    rel_index = []

    flag = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            else:
                flag += 1
                head, tail, rel = line.strip().split(" ")[:3]
                edge_index.append([int(head), int(tail)])
                rel_index.append(int(rel))

        f.close()
    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(len(list(set(rel_index))))

    return num_node, edge_index, rel_index, num_rel

def read_interactions(path, drug_dict):
    interactions = []
    all_drug_in_ddi = []
    positive_drug_inter_dict = {}
    # positive_num = 0
    # negative_num = 0
    for pt in ['train', 'valid_S0', 'valid_S1', 'valid_S2', 'test_S0', 'test_S1', 'test_S2']:
        with open(path + pt + '.txt', 'r') as f:
            for line in f.readlines():
                drug1_id, drug2_id, label = line.strip().split(" ")[:3]
                if drug1_id in drug_dict and drug2_id in drug_dict:
                    all_drug_in_ddi.append(drug1_id)
                    all_drug_in_ddi.append(drug2_id)
                    # if float(label) > 0:
                    #     positive_num += 1
                    # else:
                    #     negative_num += 1
                    if drug1_id in positive_drug_inter_dict:
                        if drug2_id not in positive_drug_inter_dict[drug1_id]:
                            positive_drug_inter_dict[drug1_id].append(drug2_id)
                            interactions.append([int(drug1_id), int(drug2_id), int(label)])
                    else:
                        positive_drug_inter_dict[drug1_id] = [drug2_id]
                        interactions.append([int(drug1_id), int(drug2_id), int(label)])
            f.close()

    return np.array(interactions, dtype=int), set(all_drug_in_ddi)

def generate_node_subgraphs(dataset, drug_id, network_edge_index, network_rel_index, num_rel, args):

    method = "randomWalk"
    edge_index = torch.from_numpy(np.array(network_edge_index).T) ##[2, num_edges]
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row),0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index),1)

    paths = "data/" + str(dataset) + "/" + str(method) + "/"

    if not os.path.exists(paths):
        # os.mkdir(paths)
        os.makedirs(paths)

    # subgraphs, max_degree, max_rel_num = rwExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
    #                                                     sub_num=args.graph_fixed_num, length=args.fixed_num)
    subgraphs, max_degree, max_rel_num = rwExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel, sub_num=1, length=32)

    return subgraphs, max_degree, max_rel_num

def rwExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, sub_num, length):

    json_path = shortest_paths + "rw_num_" + str(sub_num) + "_length_" + str(length) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_index.transpose(1,0).numpy().tolist())
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    exist_set = np.unique(np.array(edge_index)).tolist()
    for d in drug_id:
        if int(d) in exist_set:
            subsets = Node2vec(start_nodes=[int(d)], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() ##返回一个list
            mapping_id = subsets.index(int(d))
            mapping_list = [False for _ in range(len((subsets)))]
            mapping_list[mapping_id] = True
            sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
            row_sub, col_sub = sub_edge_index
            ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_value = [1 for _ in range(len(new_s_edge_index))]
            new_s_rel = sub_rel_index.numpy().tolist()

            s_edge_index = new_s_edge_index.copy()
            s_value = new_s_value.copy()
            s_rel = new_s_rel.copy()

            edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
            sp_edge_index = edge_index_value[:, :2]
            sp_value = edge_index_value[:, 2]
        else:
            subsets = [int(d)]
            mapping_list = [True]

            sub_edge_index, sub_rel_index = torch.tensor([[0],[0]]), torch.tensor([23])
            row_sub, col_sub = sub_edge_index
            ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_value = [1 for _ in range(len(new_s_edge_index))]
            new_s_rel = sub_rel_index.numpy().tolist()

            s_edge_index = new_s_edge_index.copy()
            s_value = new_s_value.copy()
            s_rel = new_s_rel.copy()

            edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
            sp_edge_index = edge_index_value[:, :2]
            sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item()
        # else:
        #     subsets = [int(d)]
        #     mapping_list = [True]

    ### update when checked
    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def get_label_ddi(label, num_rel):
    # y = np.zeros([self.p.num_rel * 2], dtype=np.float32)
    y = np.zeros([label.shape[0], num_rel], dtype=np.float32)
    for j in range(label.shape[0]): y[j, label[j]] = 1.0
    return torch.FloatTensor(y)

### part for CDAN

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
    
    def device(self, de):
        self.random_matrix = [val.to(de) for val in self.random_matrix]


### loss function part

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def CDAN(input_list, ad_net, device, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
