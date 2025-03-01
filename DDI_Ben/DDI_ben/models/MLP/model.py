import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class MLP(nn.Module):
    def __init__(self, num_ent, num_rel, nhid, args, init_feat = None):
        super(MLP, self).__init__()
        self.args = args
        if args.use_feat:
            self.entity_embedding = nn.Parameter(init_feat, requires_grad=False)
            self.lin1 = nn.Linear(init_feat.size(1)*2, nhid)
        else:
            self.entity_embedding  = nn.Parameter(torch.zeros(num_ent, nhid), requires_grad=False)
            nn.init.xavier_normal_(tensor=self.entity_embedding)
            self.lin1 = nn.Linear(nhid*2, nhid)
        self.dropout = nn.Dropout(args.mlp_dropout)
        self.lin2 = nn.Linear(nhid, num_rel)
        if self.args.dataset == 'drugbank':
            self.bceloss	= torch.nn.BCELoss()
        elif self.args.dataset == 'twosides':
            # self.bceloss	= torch.nn.BCELoss()
            self.bceloss	= torch.nn.BCELoss(weight = torch.tensor(args.loss_weight))

        self.cdan_dim = nhid
        
    def loss(self, pred, true_label):
        # return self.bceloss(pred, true_label)
        if self.args.dataset == 'drugbank':
            return self.bceloss(torch.softmax(pred,1), true_label)
        elif self.args.dataset == 'twosides':
            return self.bceloss(torch.sigmoid(pred)*true_label[:,:-1], true_label[:,:-1]*true_label[:,-1].unsqueeze(1))

    def forward(self, data):
        x = torch.cat([self.entity_embedding[data[0]], self.entity_embedding[data[1]]], dim=1)
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.softmax(x, dim=1)
        y = self.dropout(x)
        final_layer = y
        x = self.lin2(y)
        if self.args.dataset == 'drugbank':
            x = F.relu(x)
        # elif self.args.dataset == 'twosides':
        #     x = torch.sigmoid(x)
        # x = F.softmax(x, dim=1)
        if self.args.adversarial:
            return x, final_layer
        else:
            return x
