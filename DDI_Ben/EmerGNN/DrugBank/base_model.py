import torch
import numpy as np

import torch.nn
import torch.nn.functional as F

from utils import batch_by_size
from tqdm import tqdm
from torch.optim import Adam
from models import EmerGNN
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import loss as loss_func
import wandb


class BaseModel(object):
    def __init__(self, eval_ent, eval_rel, args, entity_vocab=None, relation_vocab=None):
        self.model = EmerGNN(eval_ent, eval_rel, args)
        if args.load_model:
            adversarial_suffix = '_adv' if args.adversarial else ''
            state_dict = torch.load(args.dataset+'_saved_model'+adversarial_suffix+'.pt')
            self.model.load_state_dict(state_dict)
        self.model.cuda()

        self.eval_ent = eval_ent
        self.eval_rel = eval_rel
        self.all_rel = args.all_rel
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max')

        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab

        if args.adversarial:
            self.optimizer_ad = Adam(self.model.ad_net.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
            self.scheduler_ad = ReduceLROnPlateau(self.optimizer_ad, mode='max')
            self.model.random_layer.cuda()
            self.model.ad_net.cuda()

    def train(self, train_pos, train_neg, train1_pos, train1_neg, KG):
        if self.args.adversarial:
            head, tail, label = train_pos[:, 0], train_pos[:, 1], train_pos[:, 2]
            head1, tail1, label1 = train1_pos[:, 0], train1_pos[:, 1], train1_pos[:, 2]
            n_train = len(head)
            n_train1 = len(head1)
            n_batch = self.args.n_batch
            target_len = max(n_train, n_train1)
            repeat_factor_head = -(-target_len // n_train)
            repeat_factor_head1 = -(-target_len // n_train1)
            head, tail, label = head.repeat(repeat_factor_head)[:target_len], \
                                tail.repeat(repeat_factor_head)[:target_len], \
                                label.repeat(repeat_factor_head)[:target_len]
            head1, tail1, label1 = head1.repeat(repeat_factor_head1)[:target_len], \
                                tail1.repeat(repeat_factor_head1)[:target_len], \
                                label1.repeat(repeat_factor_head1)[:target_len]
            perm = torch.randperm(target_len).cuda()
            head1, tail1, label1 = head1[perm], tail1[perm], label1[perm]

            loss_epoch = 0
            self.model.train()
            for h, t, r, h1, t1, r1 in tqdm(batch_by_size(n_batch, head, tail, label, head1, tail1, label1, n_sample=n_train),
                ncols=100, leave=False, total=len(head)//n_batch+int(len(head)%n_batch>0)):
                self.model.zero_grad()
                self.optimizer.zero_grad()
                self.optimizer_ad.zero_grad()
                ht_embed = self.model.enc_ht(h, t, KG)
                ht_embed_adv = self.model.enc_ht(h1, t1, KG)
                scores = self.model.enc_r(ht_embed)
                scores_adv = self.model.enc_r(ht_embed_adv)
                rela_scores = torch.nn.Softmax(dim=1)(scores)
                rela_scores_adv = torch.nn.Softmax(dim=1)(scores_adv)
                final_layer_comb = torch.cat([ht_embed, ht_embed_adv], 0)
                pred_comb = torch.cat([rela_scores, rela_scores_adv], 0)
                p_score = scores[torch.arange(len(r)).cuda(), r]
                n_score = scores
                max_n = torch.max(n_score, 1, keepdim=True)[0]
                loss = -p_score + max_n + torch.log(torch.sum(torch.exp(n_score - max_n), 1))
                num_elements = loss.numel()
                loss = loss.sum()
                ad_loss = loss_func.CDAN([final_layer_comb, pred_comb], self.model.ad_net, None, None, self.model.random_layer)*num_elements*self.args.adversarial_weight
                print(f" | Classification Loss: {loss.item()} | Adversarial Loss: {ad_loss.item()} | Total loss: {loss.item() + ad_loss.item()}")
                wandb.log({"train_loss": loss.item()})
                wandb.log({"train_ad_loss": ad_loss.item()})
                loss += ad_loss
                wandb.log({"train_total_loss": loss.item()})

                loss.backward()
                self.optimizer.step()
                self.optimizer_ad.step()
                loss_epoch += loss.item()

        else:
            head, tail, label = train_pos[:,0], train_pos[:,1], train_pos[:,2]
            n_train = len(head)
            n_batch = self.args.n_batch

            loss_epoch = 0
            self.model.train()
            for h, t, r in tqdm(batch_by_size(n_batch, head, tail, label, n_sample=n_train),
                ncols=100, leave=False, total=len(head)//n_batch+int(len(head)%n_batch>0)):
                self.model.zero_grad()
                ht_embed = self.model.enc_ht(h, t, KG)
                scores = self.model.enc_r(ht_embed)
                p_score = scores[torch.arange(len(r)).cuda(), r]
                n_score = scores
                max_n = torch.max(n_score, 1, keepdim=True)[0]
                loss = -p_score + max_n + torch.log(torch.sum(torch.exp(n_score - max_n), 1))
                loss = loss.sum()

                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()


    def evaluate(self, test_pos, test_neg, KG):
        heads, tails, relas = test_pos[:,0], test_pos[:,1], test_pos[:,2]
        batch_size = self.args.test_batch_size
        num_batch = len(heads) // batch_size + int(len(heads)%batch_size>0)

        rela_probs = []
        self.model.eval()
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, len(heads))
            batch_h = heads[start:end].cuda()
            batch_t = tails[start:end].cuda()
            ht_embed = self.model.enc_ht(batch_h, batch_t, KG)
            scores = self.model.enc_r(ht_embed)
            rela_scores = F.softmax(scores, dim=-1).data.cpu().numpy()

            rela_probs.append(rela_scores)
        rela_probs = np.concatenate(rela_probs)
        pred = np.argmax(rela_probs, axis=1)
        label = relas.data.cpu().numpy()
        accuracy = np.sum(pred == label) / len(pred)
        f1 = f1_score(label, pred, average='macro')
        kappa = cohen_kappa_score(label, pred)

        cm = confusion_matrix(label, pred, labels=range(86))
        accuracy_per_class = np.diagonal(cm) / cm.sum(axis=1)

        return f1, accuracy, kappa, accuracy_per_class
        # return f1, accuracy, kappa

    def test_single(self, triplet, KG):
        heads = triplet[0].unsqueeze(0)
        tails = triplet[1].unsqueeze(0)
        ht_embed = self.model.enc_ht(heads, tails, KG)
        scores = self.model.enc_r(ht_embed)
        rela_scores = F.softmax(scores, dim=-1).data.cpu().numpy()

        pred = np.argmax(rela_scores, axis=-1)
        return pred[0]


    def visualize(self, triplet, KG, head_batch=True):
        h, t, r = triplet[0], triplet[1], triplet[2:]
        paths, weights = self.model.visualize_forward(h.unsqueeze(0), t.unsqueeze(0), r.unsqueeze(0), KG, 5, head_batch)
        outputs = []
        rel_weights = [0] * (self.all_rel - self.eval_rel)
        rel_freq = [0] * self.all_rel
        for path, weight in zip(paths, weights):
            out_str = '%4f\t' % weight
            for i in range(len(path)):
                h, t, r = path[i]
                h_name = self.entity_vocab[h]
                t_name = self.entity_vocab[t]
                if r == 2*self.all_rel:
                    r_name = 'idd'
                else:
                    r_mod = r % self.all_rel
                    if r_mod >= self.eval_rel:
                        r_name = self.relation_vocab[r_mod]
                    else:
                        r_name = str(r % self.all_rel)
                    rel_freq[r_mod] += 1

                if r >= self.all_rel and r < 2*self.all_rel:
                    r_name += "_inv"
                if r >= self.eval_rel and r < self.all_rel:
                    rel_weights[r-self.eval_rel] += 1
                elif r >= self.all_rel+self.eval_rel and r < 2*self.all_rel:
                    rel_weights[r-self.eval_rel-self.all_rel] += 1

                if i == 0:
                    out_str += '< %s, %6s, %18s' % (h_name, r_name, t_name)
                else:
                    out_str += ', %6s, %18s' % (r_name, t_name)
            out_str += ' >\n'
            outputs.append(out_str)
        return outputs, np.array(rel_weights), np.array(rel_freq)

    def KG_relation_weights(self, triplets, KG):
        heads, tails = triplets[:,0], triplets[:,1]
        batch_size = self.args.test_batch_size
        num_batch = len(heads) // batch_size + int(len(heads)%batch_size>0)
        rel_weights = [[] for i in range(self.args.length)]
        self.model.eval()
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, len(heads))
            batch_h = heads[start:end].cuda()
            batch_t = tails[start:end].cuda()
            relations = self.model.get_attention_weights(batch_h, batch_t, KG)
            for l in range(self.args.length):
                rel_weights[l].append(relations[l])
       
        all_weights = 0
        for l in range(self.args.length):
            rel_weight = np.concatenate(rel_weights[l], axis=0) # [N, n_rel]
            rel_weight = np.mean(rel_weight, axis=0)    # n_rel
            kg_weight = rel_weight[self.eval_rel:self.all_rel]
            kg_weight += rel_weight[self.all_rel+self.eval_rel:2*self.all_rel]
            kg_weight /= 2
            all_weights += kg_weight
            print(l, list(np.round(kg_weight, 2)))
        print(list(np.round(all_weights/self.args.length, 2)))

    def save_model(self, out_str=''):
        adversarial_suffix = '_adv' if self.args.adversarial else ''
        torch.save(self.model.state_dict(), self.args.dataset+'_saved_model'+adversarial_suffix+'.pt')
        print(out_str, 'model saved')
