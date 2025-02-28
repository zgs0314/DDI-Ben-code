import os
from torch.utils.data import Dataset
from utils import *
from ordered_set import OrderedSet
from collections import defaultdict as ddict
import torch
from torch.utils.data import DataLoader 
import scipy.sparse as sp
from itertools import combinations
import json

num_ent = {'drugbank': 1710, 'twosides': 645, 'HetioNet': 34124}
num_rel = {'drugbank': 86, 'twosides': 209} # 209, 309

class Data_record():
    def __init__(self, args, emb_return):
        self.args = args
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        #     self.devide = torch.device('cpu')

        if args.bionet == 'HetioNet':
            self.BN_path = 'initial/{}/relations_2hop'.format(args.dataset)
        elif args.bionet == 'PrimeKG':
            self.BN_path = 'initial/{}/BN_Primekg'.format(args.dataset)

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

        self.link_aug_num = 0

        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        self.triplets = load_data(args)
        # self.triplets_all = self.triplets['train'] + self.triplets['valid'] + self.triplets['test']

        self.data = ddict(list)
        sr2o = ddict(set)

        self.link_aug_num = 0
        if args.data_aug:
            # self.selected_link = [0,1,2,5,6,7,8,9,10,13,15,17,18,19,21]
            self.selected_link = [1,5,6,7,10,15,18,21]
            self.link_aug_num = len(self.selected_link)
            trip_aug = []
            file = open('./data/{}.txt'.format(self.BN_path))
            for j in file:
                str_lin = j.strip().split(' ')
                trip = [int(j) for j in str_lin]
                if trip[2] in self.selected_link:
                    trip_aug.append([trip[0], trip[1], trip[2]])
            aug_num_begin = num_ent[args.dataset]
            aug_ent_in = np.unique(np.array(trip_aug).flatten())
            ind_dict = {}
            rel_dict = {self.selected_link[j]:num_rel[args.dataset] + j for j in range(len(self.selected_link))}
            for j in aug_ent_in:
                if j >= num_ent[args.dataset]:
                    ind_dict[j] = aug_num_begin
                    aug_num_begin += 1
                else:
                    ind_dict[j] = j
            for j in trip_aug:
                j = [ind_dict[j[0]], ind_dict[j[1]], rel_dict[j[2]]]
                self.triplets['train'].append(j)

        if args.use_reverse_edge:
            self.num_rel, self.args.num_rel = (num_rel[args.dataset] + self.link_aug_num) * 2, (num_rel[args.dataset] + self.link_aug_num) * 2
        else:
            self.num_rel, self.args.num_rel = num_rel[args.dataset] + self.link_aug_num, num_rel[args.dataset] + self.link_aug_num
        
        if args.data_aug:
            self.num_ent, self.args.num_ent = aug_num_begin, aug_num_begin ### 31130
        else:
            self.num_ent, self.args.num_ent = num_ent[args.dataset], num_ent[args.dataset]

        self.include_splits = list(self.triplets.keys())
        self.split_not_train = [j for j in self.include_splits if j != 'train']
        
        if args.use_reverse_edge:
            index_plus = int(self.args.num_rel/2)
        
        for split in self.include_splits:
            if split == 'train' and self.args.model in ['ComplEx', 'MSTE'] and args.dataset == 'twosides': 
                for j in range(int(len(self.triplets[split])/2)):
                    sub, obj, rel, neg_add = self.triplets[split][j*2][0], self.triplets[split][j*2][1], np.where(np.array(self.triplets[split][j*2][2])[:-1]==1)[0], [self.triplets[split][j*2+1][0], self.triplets[split][j*2+1][1]]
                    for k in rel:
                        self.data[split].append((sub, obj, [neg_add[0], neg_add[1], k]))
                        sr2o[(sub, obj)].add((neg_add[0], neg_add[1], k))
            else:
                if split in ['valid_S1', 'test_S1'] and self.args.model in ['ComplEx', 'MSTE'] and self.args.dataset == 'twosides' and self.args.adversarial:
                    for j in range(int(len(self.triplets[split])/2)):
                        sub, obj, rel, neg_add = self.triplets[split][j*2][0], self.triplets[split][j*2][1], np.where(np.array(self.triplets[split][j*2][2])[:-1]==1)[0], [self.triplets[split][j*2+1][0], self.triplets[split][j*2+1][1]]
                        for k in rel:
                            self.data[split + '_adv'].append((sub, obj, [neg_add[0], neg_add[1], k]))
                            # sr2o[(sub, obj)].add((neg_add[0], neg_add[1], k))
                for j in self.triplets[split]:
                    # sub, obj, rel = self.ent2id[j[0]], self.ent2id[j[1]], self.rel2id[str(j[2])]  
                    sub, obj, rel = j[0], j[1], j[2]
                    self.data[split].append((sub, obj, rel))

                    if split == 'train': 
                        if self.args.model in ['CompGCN', 'SkipGNN', 'Decagon'] and args.dataset == 'twosides':
                            self.true_data = self.data[split]
                        sr2o[(sub, obj)].add(rel)
                        if args.use_reverse_edge:
                            # sr2o[(obj, sub)].add(rel+index_plus)
                            self.data[split].append((obj,sub , rel+index_plus))
        
        if args.use_feat or args.model == 'CSMDDI':
            self.feat = torch.FloatTensor(np.array(load_feature(args))).to(self.device)
            self.feat_dim = self.feat.shape[1]
        else:
            if args.model == 'SkipGNN':
                features = np.eye(self.num_ent)
                self.feat = normalize(features)
                self.feat = torch.FloatTensor(features)
            elif args.model == 'KGDDI':
                self.feat = emb_return[:num_ent[args.dataset]]
            else:
                self.feat = 0
        
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.data = dict(self.data)

        for split in self.split_not_train:
            for sub, obj, rel in self.data[split]:
                sr2o[(sub, obj)].add(rel)
                # sr2o[(obj, sub)].add(rel+self.p.num_rel)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        ### train triples
        if self.args.dataset == 'twosides' and self.args.model in ['ComplEx', 'MSTE']:
            for sub, obj, rel in self.data['train']:
                self.triples['train'].append({'triple':(sub, obj, -1), 'label': rel, 'sub_samp': 1})
            if self.args.adversarial:
                for split in ['valid_S1_adv', 'test_S1_adv']:
                    for sub, obj, rel in self.data[split]:
                        self.triples[split].append({'triple':(sub, obj, -1), 'label': rel, 'sub_samp': 1})
        else:
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        ### valid & test triplets
        for split in self.split_not_train:
            for sub, obj, rel  in self.data[split]:
                self.triples[split].append({'triple': (sub, obj, rel), 	   'label': self.sr2o_all[(sub, obj)]})

        self.triples = dict(self.triples)

        if args.model == 'CompGCN':
            self.p = args
            self.p.embed_dim	= self.p.comp_dim

            ### remark: see whether we need the reverse links
			
            self.edge_index, self.edge_type = [], []
            for sub, obj, rel in self.data['train']:
                if args.dataset == 'twosides' and rel[-1] == 0:
                    continue
                self.edge_index.append((sub, obj))
                self.edge_type.append(rel)

            self.edge_index	= torch.LongTensor(self.edge_index).to(self.device).t()
            if args.dataset == 'drugbank':
                self.edge_type	= torch.LongTensor(self.edge_type).to(self.device)
            elif args.dataset == 'twosides':
                self.edge_type	= torch.LongTensor(self.edge_type)[:,:-1].to(self.device)

            if args.data_aug and args.use_feat:
                feat = torch.zeros((self.num_ent, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat

        elif args.model == 'SkipGNN':
            if args.dataset == 'drugbank':
                self.num_skipnode = 1710
                self.link = np.array(self.triplets['train'])[:,:2]
            elif args.dataset == 'twosides':
                self.num_skipnode = self.num_ent
                self.link = np.array([[j[0], j[1]] for j in self.triplets['train'] if j[2][-1] == 1])
            adj = sp.coo_matrix((np.ones(self.link.shape[0]), (self.link[:, 0], self.link[:, 1])),
                    shape=(self.num_skipnode, self.num_skipnode),
                    dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            #create skip graph
            adj2 = adj.dot(adj)
            adj2 = adj2.sign()
            adj2 = normalize_adj(adj2)
            self.adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
            adj = adj + sp.eye(adj.shape[0])
            #normalize original graph
            adj = normalize_adj(adj)
            self.adj = torch.FloatTensor(np.array(adj.todense()))

            if args.data_aug and args.use_feat:
                feat = torch.zeros((self.num_ent, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat

        elif args.model == 'CSMDDI':
            self.adj_matrix = np.zeros((num_rel[args.dataset], num_ent[args.dataset], num_ent[args.dataset]))
            np_tri = np.array(self.triplets['train'])
            # self.adj_matrix[np_tri[:,2], np_tri[:,0], np_tri[:,1]] += 1
            # self.adj_matrix[np_tri[:,2], np_tri[:,1], np_tri[:,0]] += 1
            if args.dataset == 'drugbank':
                self.adj_matrix[np_tri[:,2], np_tri[:,0], np_tri[:,1]] += 1
                self.adj_matrix[np_tri[:,2], np_tri[:,1], np_tri[:,0]] += 1
            elif args.dataset == 'twosides':
                for j in np_tri:
                    self.adj_matrix[:, j[0], j[1]] = np.array(np_tri[0][2]).astype('float64')[:-1]
                    self.adj_matrix[:, j[1], j[0]] = np.array(np_tri[0][2]).astype('float64')[:-1]
            self.train_set = []
            file = open('./data/{}/train_set.txt'.format(args.dataset))
            for j in file:
                self.train_set.append(int(j.strip()))
            self.val_test_set = list(set(range(num_ent[args.dataset])) - set(self.train_set))
            x = 0
        elif args.model == 'HINDDI':
            flag = 0
            if args.with_1hop:
                pathh = './data/{}/metapath/meta_with.npy'.format(args.dataset)
                if not os.path.exists('./data/{}/metapath/meta_with.npy'.format(args.dataset)):
                    flag = 1
            else:
                pathh = './data/{}/metapath/meta_without.npy'.format(args.dataset)
                if not os.path.exists('./data/{}/metapath/meta_without.npy'.format(args.dataset)):
                    flag = 1
            if flag:
                hop_triple = []
                selected_link = [0,1,2,5,6,7,8,9,10,13,14,15,16,17,18,19,21]
                file = open('./data/{}.txt'.format(self.BN_path))
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    if trip[2] in selected_link:
                        hop_triple.append(trip)
                if args.with_1hop:
                    for j in self.triplets['train']:
                        hop_triple.append(j)
                if args.dataset == 'twosides':
                    entity_type = [0 for j in range(35964)]
                elif args.dataset == 'drugbank':
                    entity_type = [0 for j in range(num_ent[args.bionet])]
                for i in range(num_ent[args.dataset]):
                    entity_type[i] = 'drug'
                if args.dataset == 'twosides':
                    with open('./data/initial/{}/allname2id.json'.format(args.dataset), 'r') as file:
                        data = json.load(file)
                elif args.dataset == 'drugbank':
                    with open('./data/initial/{}/entity_drug.json'.format(args.dataset), 'r') as file:
                        data = json.load(file)
                for j in data:
                    if 'Gene' in j:
                        entity_type[data[j]] = 'gene'
                    elif 'Anatomy' in j:
                        entity_type[data[j]] = 'anatomy'
                    elif 'Disease' in j:
                        entity_type[data[j]] = 'disease'
                    elif 'Symptom' in j:
                        entity_type[data[j]] = 'symptom'
                    elif 'Compound' in j or 'CID' in j:
                        entity_type[data[j]] = 'drug'
                    elif 'Pathway' in j:
                        entity_type[data[j]] = 'pathway'
                    elif 'Molecular Function' in j:
                        entity_type[data[j]] = 'molecular function'
                    elif 'Cellular Component' in j:
                        entity_type[data[j]] = 'cellular component'
                    elif 'Side Effect' in j:
                        entity_type[data[j]] = 'side effect'
                    elif 'Pharmacologic Class' in j:
                        entity_type[data[j]] = 'pharmacologic class'
                entity_class_no = ['pathway', 'symptom', 'anatomy', 'cellular component', 'molecular function']
                link_record = {}
                for j in hop_triple:
                    if entity_type[j[0]] in entity_class_no or entity_type[j[1]] in entity_class_no:
                        continue
                    if j[0] in link_record:
                        link_record[j[0]].append(j[1])
                    else:
                        link_record[j[0]] = [j[1]]
                    if j[1] in link_record:
                        link_record[j[1]].append(j[0])
                    else:
                        link_record[j[1]] = [j[0]]
                meta_matrix = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 14))
                meta_path_dict = {'drug':1, 'disease': 2, 'gene': 3, 'pharmacologic class': 4, 'side effect': 5,
                                'drug_drug': 6, 'drug_disease': 7, 'drug_gene': 8, 'drug_pharmacologic class': 9, 'drug_side effect': 10, 'gene_gene': 11, 'gene_disease': 12, 'disease_disease': 13, 
                                'disease_drug': 7, 'gene_drug': 8, 'pharmacologic class_drug': 9, 'side effect_drug': 10, 'disease_gene': 12}
                for i in range(num_ent[args.dataset]):
                    if i in link_record: ### begin
                        for j in link_record[i]:
                            if j < num_ent[args.dataset]:
                                meta_matrix[i,j,0] += 1
                            if j in link_record:
                                for k in link_record[j]:
                                    if k < num_ent[args.dataset]:
                                        meta_matrix[i,k,meta_path_dict[entity_type[j]]] += 1
                                    if k in link_record:
                                        for l in link_record[k]:
                                            if l < num_ent[args.dataset]:
                                                meta_matrix[i,l,meta_path_dict[entity_type[j] + '_' + entity_type[k]]] += 1
                    print(i)
                meta_feature = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 14 * 4))
                for j in range(14):
                    meta_feature[:,:,j*4] = meta_matrix[:,:,j]
                    meta_feature[:,:,j*4+1] = 2 * meta_matrix[:,:,j]/np.maximum((meta_matrix[:,:,j].sum(0) + (meta_matrix[:,:,j].sum(1))[None,:].T), np.ones((num_ent[args.dataset], num_ent[args.dataset])))
                    meta_feature[:,:,j*4+2] = meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(0), np.ones(num_ent[args.dataset]))
                    meta_feature[:,:,j*4+3] = meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(0), np.ones(num_ent[args.dataset])) + meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(1), np.ones((num_ent[args.dataset], 1)))
                self.meta_feature = meta_feature
                np.save(pathh, meta_feature)
            else:
                self.meta_feature = np.load(pathh)
            self.meta_feature = torch.FloatTensor(self.meta_feature)
        elif args.model == 'Decagon':
            self.edge_index = []
            if args.dataset == 'twosides':
                for sub, obj, rel in self.data['train']:
                    if rel[-1] == 1:
                        self.edge_index.append([sub, obj])
            else:
                for sub, obj, rel in self.data['train']:
                    self.edge_index.append([sub, obj])
            trip_de = []
            file = open('./data/{}.txt'.format(self.BN_path))
            if args.bionet == 'HetioNet':
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    # if trip[2] in [1,21]:
                    if trip[2] in [1, 5, 6, 7, 10, 18]:
                        trip_de.append([trip[0], trip[1]])
            elif args.bionet == 'PrimeKG':
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    # if trip[2] in [1,21]:
                    if trip[2] in [1,2,3,4,5,6,7,8]:
                        trip_de.append([trip[0], trip[1]])
            num_begin = num_ent[args.dataset]
            ent_in = np.unique(np.array(trip_de).flatten())
            ind_dict = {}
            for j in ent_in:
                if j >= num_ent[args.dataset]:
                    ind_dict[j] = num_begin
                    num_begin += 1
                else:
                    ind_dict[j] = j
            for j in trip_de:
                j = [ind_dict[j[0]], ind_dict[j[1]]]
                self.edge_index.append(j)

            self.edge_index	= torch.LongTensor(self.edge_index).to(self.device).t()

            if args.use_feat:
                feat = torch.zeros((num_begin, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat
        elif args.model == 'TIGER':
            if args.dataset == 'twosides':
                with open('./data/initial/twosides/cid2id.json', 'r') as file:
                    cid2id = json.load(file)
                with open('./data/initial/twosides/cid2smiles.json', 'r') as file:
                    cid2smiles = json.load(file)
                TG_id2smiles = {str(cid2id[j]):cid2smiles[j] for j in cid2smiles}
                print("load drug smiles graphs!!")
                TG_smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph('data/{}'.format(folder_name), TG_id2smiles)
                print("load networks !!")
                num_node, network_edge_index, network_rel_index, TG_num_rel = read_network('data/initial/twosides/relations_2hop.txt'.format(folder_name))
                print("load DDI samples!!")
                TG_labels = 0
                TG_interactions = np.concatenate([np.array([j[:2] for j in self.triplets[k]]) for k in self.include_splits])
                all_contained_drugs = np.unique(TG_interactions)
                all_contained_drugs = set([str(j) for j in all_contained_drugs])
                print("generate subgraphs!!")
                TG_drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(folder_name, all_contained_drugs,network_edge_index, network_rel_index,TG_num_rel, args)

                TG_data_sta = {
                    'num_nodes': num_node + 1,
                    'num_rel_mol': num_rel_mol_update + 1,
                    'num_rel_graph': num_rel_update + 1,
                    'num_interactions': len(TG_interactions),
                    'num_drugs_DDI': len(all_contained_drugs),
                    'max_degree_graph': max_smiles_degree + 1,
                    'max_degree_node': int(max_subgraph_degree)+1
                }

                print(TG_data_sta)
                self.TG_interactions = TG_interactions
                self.TG_labels = TG_labels
                self.TG_smile_graph = TG_smile_graph
                self.TG_drug_subgraphs = TG_drug_subgraphs
                self.TG_data_sta = TG_data_sta

            elif args.dataset == 'drugbank':
                # with open('./data/initial/drugbank/DB2SMILES_2.json', 'r') as file:
                #     TG_db2SMILES = json.load(file)
                with open('data/initial/drugbank/DB_molecular_feats.pkl', 'rb') as f:
                    x = pkl.load(f, encoding='utf-8')
                TG_id2smiles = {str(j): x['SMILES'][j] for j in range(1710)}
                for j in [   6,  136,  889, 1171, 1239, 1254]:
                    TG_id2smiles[str(j)] = ''
                # with open('./data/initial/drugbank/node2id.json', 'r') as file:
                #     TG_node2id = json.load(file)
                # TG_id2smiles = {str(TG_node2id[j]):TG_db2SMILES[j] for j in TG_db2SMILES}
                print("load drug smiles graphs!!")
                TG_smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph('data/{}'.format(folder_name), TG_id2smiles)
                print("load networks !!")
                num_node, network_edge_index, network_rel_index, TG_num_rel = read_network('data/initial/drugbank/relations_2hop.txt'.format(folder_name))
                print("load DDI samples!!")
                ### this part remain (need to be simplified)
                # interactions_label, all_contained_drugs = read_interactions('data/{}/'.format(args.dataset), TG_smile_graph)
                # self.include_splits
                TG_triplet_all = np.concatenate([np.array(self.triplets[j]) for j in self.include_splits])
                TG_interactions = TG_triplet_all[:, :2]
                TG_labels = TG_triplet_all[:, 2]
                all_contained_drugs = np.unique(TG_interactions)
                all_contained_drugs = set([str(j) for j in all_contained_drugs])
                print("generate subgraphs!!")
                TG_drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(folder_name, all_contained_drugs,network_edge_index, network_rel_index,TG_num_rel, args)

                TG_data_sta = {
                    'num_nodes': num_node + 1,
                    'num_rel_mol': num_rel_mol_update + 1,
                    'num_rel_graph': num_rel_update + 1,
                    'num_interactions': len(TG_interactions),
                    'num_drugs_DDI': len(all_contained_drugs),
                    'max_degree_graph': max_smiles_degree + 1,
                    'max_degree_node': int(max_subgraph_degree)+1
                }

                print(TG_data_sta)
                self.TG_interactions = TG_interactions
                self.TG_labels = TG_labels
                self.TG_smile_graph = TG_smile_graph
                self.TG_drug_subgraphs = TG_drug_subgraphs
                self.TG_data_sta = TG_data_sta

        ### the main part
        self.data_iter = {}
        if args.model == 'TIGER':
            if args.dataset == 'drugbank':
                self.data_iter['train'] = DataLoader(DTADataset(x=np.array(self.triplets['train'])[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
                if args.adversarial:
                    if args.adversarial_mode == 0:
                        copy_triplets = ((int(len(self.triplets['train'])/len(self.triplets['valid_' + args.adversarial_task])) + 1) * self.triplets['valid_' + args.adversarial_task])[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 3:
                        triplets_adv = self.triplets['valid_' + args.adversarial_task] + self.triplets['test_' + args.adversarial_task]
                        copy_triplets = ((int(len(self.triplets['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 2:
                        copy_triplets = ((int(len(self.triplets['train'])/len(self.triplets['adv_2'])) + 1) * self.triplets['adv_2'])[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 1:
                        copy_triplets = ((int(len(self.triplets['train'])/len(self.triplets['adv_1'])) + 1) * self.triplets['adv_1'])[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                for j in self.split_not_train:
                    self.data_iter[j] = DataLoader(DTADataset(x=np.array(self.triplets[j])[:,:2], y=np.array(self.triplets[j])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
            elif args.dataset == 'twosides':
                self.data_iter['train'] = DataLoader(DTADataset(x=np.array([k[:2] for k in self.triplets['train']]), y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
                if args.adversarial:
                    if args.adversarial_mode == 0:
                        array_new = np.array([k[:2] for k in self.triplets['valid_' + args.adversarial_task]]).tolist()
                        copy_triplets = ((int(len(self.triplets['train'])/len(self.triplets['valid_' + args.adversarial_task])) + 1) * array_new)[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 3:
                        array_new_valid = np.array([k[:2] for k in self.triplets['valid_' + args.adversarial_task]]).tolist()
                        array_new_test = np.array([k[:2] for k in self.triplets['test_' + args.adversarial_task]]).tolist()
                        triplets_adv = array_new_valid + array_new_test
                        copy_triplets = ((int(len(self.triplets['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 2:
                        triplets_adv = np.array([k[:2] for k in self.triplets['adv_2']]).tolist()
                        copy_triplets = ((int(len(self.triplets['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                    elif args.adversarial_mode == 1:
                        triplets_adv = np.array([k[:2] for k in self.triplets['adv_1']]).tolist()
                        copy_triplets = ((int(len(self.triplets['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triplets['train']))]
                        self.data_iter['train_adv'] = DataLoader(DTADataset(x=np.array(copy_triplets)[:,:2], y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) ### fake label
                for j in self.split_not_train:
                    self.data_iter[j] = DataLoader(DTADataset(x=np.array([k[:2] for k in self.triplets[j]]), y=np.array([list(k[2]) for k in self.triplets[j]]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        else:
            self.data_iter['train'] = self.get_data_loader(TrainDataset, 'train', args.batch_size)
            if args.adversarial:
                if args.adversarial_mode == 0:
                    copy_triplets = ((int(len(self.triples['train'])/len(self.triples['valid_' + args.adversarial_task])) + 1) * self.triples['valid_' + args.adversarial_task])[:int(len(self.triples['train']))]
                elif args.adversarial_mode == 3:
                    if self.args.model in ['ComplEx', 'MSTE'] and self.args.dataset == 'twosides':
                        triplets_adv = self.triples['valid_' + args.adversarial_task + '_adv'] + self.triples['test_' + args.adversarial_task + '_adv']
                    else:
                        triplets_adv = self.triples['valid_' + args.adversarial_task] + self.triples['test_' + args.adversarial_task]
                    copy_triplets = ((int(len(self.triples['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triples['train']))]
                elif args.adversarial_mode == 2:
                    triplets_adv = self.triples['adv_2']
                    copy_triplets = ((int(len(self.triples['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triples['train']))]
                elif args.adversarial_mode == 1:
                    triplets_adv = self.triples['adv_1']
                    copy_triplets = ((int(len(self.triples['train'])/len(triplets_adv)) + 1) * triplets_adv)[:int(len(self.triples['train']))]
                self.data_iter['train_adv'] = DataLoader(
                    TrainDataset(copy_triplets, self.args),
                    batch_size      = args.batch_size,
                    shuffle         = True,
                    num_workers     = 10, ### set the default numworkers to 10
                    collate_fn      = TrainDataset.collate_fn,
                    drop_last=True
                )
            for j in self.split_not_train:
                self.data_iter[j] = self.get_data_loader(TestDataset, j, args.batch_size, shuffle = False)

    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        return  DataLoader(
            dataset_class(self.triples[split], self.args),
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = 10, ### set the default numworkers to 10
            collate_fn      = dataset_class.collate_fn,
            drop_last=True
        )

class TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		# self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele	= self.triples[idx]
		if 'sub_samp' in ele:
			triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		else:
			# triple, label = torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int) 
			# [ele['triple'][0], ele['triple'][1], -1], np.array(ele['label']).astype(int)
			triple, label = torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label']).astype(int) 
		if self.p.dataset == 'drugbank': trp_label = self.get_label_ddi(label) 
		elif self.p.dataset == 'twosides': 
			label = label[0]
			trp_label = torch.FloatTensor(label)
			# if label[-1] == 1: 
			# 	trp_label = torch.FloatTensor(label)[:-1]
			# else:
			# 	trp_label = torch.zeros(self.p.num_rel)
        
		if self.p.model in ['ComplEx', 'MSTE']:
			if self.p.dataset == 'drugbank':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])
			elif self.p.dataset == 'twosides':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][2], ele['label'][0] , ele['label'][1]])
				trp_label = torch.LongTensor([ele['label'][2]])
				# trp_label = self.get_label_ddi(ele['label'][2]) 

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	# def get_neg_ent(self, triple, label):
	# 	def get(triple, label):
	# 		pos_obj		= label
	# 		mask		= np.ones([self.p.num_ent], dtype=np.bool)
	# 		mask[label]	= 0
	# 		neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
	# 		neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

	# 		return neg_ent

	# 	neg_ent = get(triple, label)
	# 	return neg_ent

	def get_label_ddi(self, label):
		# y = np.zeros([self.p.num_rel * 2], dtype=np.float32)
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		if self.p.dataset == 'drugbank': triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		elif self.p.dataset == 'twosides': triple, label	= torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label'])[0]
		if self.p.dataset == 'drugbank': label		= self.get_label_ddi(label)
		elif self.p.dataset == 'twosides': label = torch.FloatTensor(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		if self.p.use_reverse_edge:
			y = np.zeros([int(self.p.num_rel/2)], dtype=np.float32)
		else:
			y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

### dataset for tiger
from torch_geometric.data import InMemoryDataset, DataLoader, Batch, Dataset
from torch_geometric import data as DATA

class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, dt = None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.dt = dt
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2 = self.process(x, y, sub_graph, smile_graph)

    def read_drug_info(self, drug_id):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        if edge_index == 0:
            c_size = 1
            features = [[0 for j in range(67)]]
            edge_index = [[0, 0]]
            rel_index = [0]
            sp_edge_index = [[0, 0]]
            sp_value = [1]
            sp_rel = [1]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            #   y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                # y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        # labels = int(self.labels[idx])
        if self.dt == 'drugbank':
            labels = torch.LongTensor([self.labels[idx]])
        else:
            labels = torch.FloatTensor(self.labels[idx])

        # drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id, labels)
        # drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id, labels)

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id)
        drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subrgraph, labels


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])

    return batchA, batchB, batchC, batchD


            # bn_triple = []
            # hop_triple_3 = []
            # selected_link = [0,1,2,5,6,7,8,9,10,13,14,15,16,17,18,19,21]
            # file = open('./data/{}/{}.txt'.format(args.dataset, self.BN_path))
            # for j in file:
            #     str_lin = j.strip().split(' ')
            #     trip = [int(j) for j in str_lin]
            #     if trip[0] < num_ent[args.dataset] or trip[1] < num_ent[args.dataset]:
            #         bn_triple.append(trip)
            #     if trip[2] in selected_link:
            #         hop_triple_3.append(trip)
            # ### add part
            # if args.with_1hop:
            #     for j in self.triplets['train']:
            #         bn_triple.append([j[0], j[1], 1])
            #         hop_triple_3.append([j[0], j[1], 1])
            # bn_triple = np.array(bn_triple)
            # bn_1hop_linktype = np.unique(bn_triple[:,2]).tolist()
            # link_record  = {j: {} for  j in bn_1hop_linktype}
            # link_record_2 = {j: {} for  j in selected_link}
            # len_edge = num_ent[args.dataset]
            # for j in bn_triple:
            #     if j.max() < num_ent[args.dataset]:
            #         if j[0] in link_record[j[2]]:
            #             link_record[j[2]][j[0]].append(j[1])
            #         else:
            #             link_record[j[2]][j[0]] = [j[1]]
            #         if j[1] in link_record[j[2]]:
            #             link_record[j[2]][j[1]].append(j[0])
            #         else:
            #             link_record[j[2]][j[1]] = [j[0]]
            #     else:
            #         jmax, jmin = j[:2].max(), j[:2].min()
            #         if jmax in link_record[j[2]]:
            #             link_record[j[2]][jmax].append(jmin)
            #         else:
            #             link_record[j[2]][jmax] = [jmin]
            # for j in hop_triple_3:
            #     if j[0] in link_record_2[j[2]]:
            #         link_record_2[j[2]][j[0]].append(j[1])
            #     else:
            #         link_record_2[j[2]][j[0]] = [j[1]]
            #     if j[1] in link_record_2[j[2]]:
            #         link_record_2[j[2]][j[1]].append(j[0])
            #     else:
            #         link_record_2[j[2]][j[1]] = [j[0]]
            # ### one hop meta feature 
            # if not os.path.exists('./data/{}/metapath1'.format(args.dataset)):
            #     os.makedirs('./data/{}/metapath1'.format(args.dataset))
            # if not os.path.exists('./data/{}/metapath1/2hop_sym.npy'.format(args.dataset)):
            #     meta_matrix_2sym = np.zeros((num_ent[args.dataset], num_ent[args.dataset], len(bn_1hop_linktype)))
            #     for i1, i2 in enumerate(bn_1hop_linktype):
            #         for j in link_record[i2]:
            #             add_list = list(combinations(link_record[i2][j], 2))
            #             for k in add_list:
            #                 meta_matrix_2sym[k[1], k[0], i1] += 1
            #                 meta_matrix_2sym[k[0], k[1], i1] += 1
            #     meta_feature_2sym = np.zeros((num_ent[args.dataset], num_ent[args.dataset], len(bn_1hop_linktype)*4))
            #     for j in range(len(bn_1hop_linktype)):
            #         meta_feature_2sym[:,:,j*4] = meta_matrix_2sym[:,:,j]
            #         meta_feature_2sym[:,:,j*4+1] = 2 * meta_matrix_2sym[:,:,j]/np.maximum((meta_matrix_2sym[:,:,j].sum(0) + (meta_matrix_2sym[:,:,j].sum(1))[None,:].T), np.ones((len_edge, len_edge)))
            #         meta_feature_2sym[:,:,j*4+2] = meta_matrix_2sym[:,:,j]/np.maximum(meta_matrix_2sym[:,:,j].sum(0), np.ones(len_edge))
            #         meta_feature_2sym[:,:,j*4+3] = meta_matrix_2sym[:,:,j]/np.maximum(meta_matrix_2sym[:,:,j].sum(0), np.ones(len_edge)) + meta_matrix_2sym[:,:,j]/np.maximum(meta_matrix_2sym[:,:,j].sum(1), np.ones((len_edge, 1)))
            #     np.save('./data/{}/metapath1/2hop_sym.npy'.format(args.dataset), meta_feature_2sym)
            # else:
            #     meta_feature_2sym = np.load('./data/{}/metapath1/2hop_sym.npy'.format(args.dataset))
            # ### specially for hetionet
            # if not os.path.exists('./data/{}/metapath1/2hop_asym.npy'.format(args.dataset)):
            #     meta_matrix_2asym = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 4))
            #     for j in link_record[5]:
            #         if j in link_record[10]:
            #             for k in link_record[5][j]:
            #                 for l in link_record[10][j]:
            #                     meta_matrix_2asym[k, l, 0] += 1
            #     for j in link_record[6]:
            #         if j in link_record[7]:
            #             for k in link_record[6][j]:
            #                 for l in link_record[7][j]:
            #                     meta_matrix_2asym[k, l, 1] += 1
            #         if j in link_record[18]:
            #             for k in link_record[6][j]:
            #                 for l in link_record[18][j]:
            #                     meta_matrix_2asym[k, l, 2] += 1
            #     for j in link_record[7]:
            #         if j in link_record[18]:
            #             for k in link_record[7][j]:
            #                 for l in link_record[18][j]:
            #                     meta_matrix_2asym[k, l, 3] += 1
            #     meta_feature_2asym = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 4 * 4))
            #     for j in range(4):
            #         meta_feature_2asym[:,:,j*4] = meta_matrix_2asym[:,:,j]
            #         meta_feature_2asym[:,:,j*4+1] = (meta_matrix_2asym[:,:,j] + meta_matrix_2asym[:,:,j].T)/np.maximum((meta_matrix_2asym[:,:,j].sum(0) + (meta_matrix_2asym[:,:,j].sum(1))[None,:].T), np.ones((len_edge, len_edge)))
            #         meta_feature_2asym[:,:,j*4+2] = meta_matrix_2asym[:,:,j]/np.maximum(meta_matrix_2asym[:,:,j].sum(0), np.ones(len_edge))
            #         meta_feature_2asym[:,:,j*4+3] = meta_matrix_2asym[:,:,j]/np.maximum(meta_matrix_2asym[:,:,j].sum(0), np.ones(len_edge)) + meta_matrix_2asym[:,:,j].T/np.maximum((meta_matrix_2asym[:,:,j].T.sum(1))[None,:].T, np.ones((len_edge, 1)))
            #     np.save('./data/{}/metapath1/2hop_asym.npy'.format(args.dataset), meta_feature_2asym)
            # else:
            #     meta_feature_2asym = np.load('./data/{}/metapath1/2hop_asym.npy'.format(args.dataset))
            # # setting = 2 ### 2,3
            # if not os.path.exists('./data/{}/metapath1/3hop.npy'.format(args.dataset)):
            #     meta_matrix_3 = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 55))
            #     meta_feature_3 = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 55 * 4))
            #     ### begin with drug
            #     # for j in link_record[1]:
            #     lastdim_index = 0
            #     for i in bn_1hop_linktype: #[1 5 6 7 10 15 18 21]
            #         for j in link_record_2[i]: ### 1-hop
            #             if j in link_record[1]:
            #                 for k in link_record_2[i][j]: ### 2-hop
            #                     if k in link_record[i]:
            #                         for l in link_record[i][k]: ### 3-hop
            #                             for m in link_record[1][j]: ### source
            #                                 meta_matrix_3[m, l, lastdim_index] += 1
            #         print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #         print(meta_matrix_3[:,:,lastdim_index].sum())
            #         print(lastdim_index)
            #         lastdim_index += 1
            #     for i in [(5,10), (10,5), (6,7), (7,6), (6,18), (18,6), (7,18), (18,7)]:
            #         for j in link_record_2[i[0]]: ### 1-hop
            #             if j in link_record[1]:
            #                 for k in link_record_2[i[0]][j]: ### 2-hop
            #                     if k in link_record[i[1]]:
            #                         for l in link_record[i[1]][k]: ### 3-hop
            #                             for m in link_record[1][j]: ### source
            #                                 meta_matrix_3[m, l, lastdim_index] += 1
            #         print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #         print(meta_matrix_3[:,:,lastdim_index].sum())
            #         print(lastdim_index)
            #         lastdim_index += 1
            #     for edge_13 in [6,7,18]:
            #         for edge_2 in [0, 13, 17]:
            #             for j in link_record_2[edge_2]: ### 1-hop
            #                 if j in link_record[edge_13]:
            #                     for k in link_record_2[edge_2][j]: ### 2-hop
            #                         if k in link_record[edge_13]:
            #                             for l in link_record[edge_13][k]: ### 3-hop
            #                                 for m in link_record[edge_13][j]: ### source
            #                                     meta_matrix_3[m, l, lastdim_index] += 1
            #             print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #             print(meta_matrix_3[:,:,lastdim_index].sum())
            #             print(lastdim_index)
            #             lastdim_index += 1
            #     for edge_13 in [(6,7), (6,18), (7,18)]:
            #         for edge_2 in [0, 13, 17]:
            #             for j in link_record_2[edge_2]: ### 1-hop
            #                 if j in link_record[edge_13[0]]:
            #                     for k in link_record_2[edge_2][j]: ### 2-hop
            #                         if k in link_record[edge_13[1]]:
            #                             for l in link_record[edge_13[1]][k]: ### 3-hop
            #                                 for m in link_record[edge_13[0]][j]: ### source
            #                                     meta_matrix_3[m, l, lastdim_index] += 1
            #             print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #             print(meta_matrix_3[:,:,lastdim_index].sum())
            #             print(lastdim_index)
            #             lastdim_index += 1
            #     for edge_13 in [(5,5), (10,10), (5,10)]:
            #         for j in link_record_2[8]: ### 1-hop
            #             if j in link_record[edge_13[0]]:
            #                 for k in link_record_2[8][j]: ### 2-hop
            #                     if k in link_record[edge_13[1]]:
            #                         for l in link_record[edge_13[1]][k]: ### 3-hop
            #                             for m in link_record[edge_13[0]][j]: ### source
            #                                 meta_matrix_3[m, l, lastdim_index] += 1
            #         print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #         print(meta_matrix_3[:,:,lastdim_index].sum())
            #         print(lastdim_index)
            #         lastdim_index += 1
            #     for edge_1 in [5,10]:
            #         for edge_2 in [2,9,19]:
            #             for edge_3 in [6,7,18]:
            #                 for j in link_record_2[edge_2]: ### 1-hop
            #                     if j in link_record[edge_1]:
            #                         for k in link_record_2[edge_2][j]: ### 2-hop
            #                             if k in link_record[edge_3]:
            #                                 for l in link_record[edge_3][k]: ### 3-hop
            #                                     for m in link_record[edge_1][j]: ### source
            #                                         meta_matrix_3[m, l, lastdim_index] += 1
            #                 print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            #                 print(meta_matrix_3[:,:,lastdim_index].sum())
            #                 print(lastdim_index)
            #                 lastdim_index += 1
            #     for j in range(55):
            #         meta_feature_3[:,:,j*4] = meta_matrix_3[:,:,j]
            #         meta_feature_3[:,:,j*4+1] = (meta_matrix_3[:,:,j] + meta_matrix_3[:,:,j].T)/np.maximum((meta_matrix_3[:,:,j].sum(0) + (meta_matrix_3[:,:,j].sum(1))[None,:].T), np.ones((len_edge, len_edge)))
            #         meta_feature_3[:,:,j*4+2] = meta_matrix_3[:,:,j]/np.maximum(meta_matrix_3[:,:,j].sum(0), np.ones(len_edge))
            #         meta_feature_3[:,:,j*4+3] = meta_matrix_3[:,:,j]/np.maximum(meta_matrix_3[:,:,j].sum(0), np.ones(len_edge)) + meta_matrix_3[:,:,j].T/np.maximum((meta_matrix_3[:,:,j].T.sum(1))[None,:].T, np.ones((len_edge, 1)))
            #     np.save('./data/{}/metapath1/3hop.npy'.format(args.dataset), meta_feature_3)
            # else:
            #     meta_feature_3 = np.load('./data/{}/metapath1/3hop.npy'.format(args.dataset))
                    
            # if args.hin_featset == 1:
            #     self.meta_feature = torch.FloatTensor(meta_feature_2sym)
            # elif args.hin_featset == 2:
            #     self.meta_feature = torch.FloatTensor(np.concatenate((meta_feature_2sym, meta_feature_2asym), axis=2))
            # elif args.hin_featset == 3:
            #     self.meta_feature = torch.FloatTensor(np.concatenate((meta_feature_2sym, meta_feature_2asym, meta_feature_3), axis=2))
            # print(np.concatenate((meta_feature_2sym, meta_feature_2asym, meta_feature_3), axis=2).shape)
