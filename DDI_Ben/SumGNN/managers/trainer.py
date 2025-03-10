import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import json
from torch.nn.utils import clip_grad_norm_
from model.cdan_model import *


class Trainer():
    def __init__(self, params, graph_classifier, train, train_evaluator = None, valid_evaluator=None, test_evaluator = None):
        self.graph_classifier = graph_classifier
        self.train_evaluator=train_evaluator
        self.valid_evaluator = valid_evaluator
        self.params = params
        if params.adversarial:
            self.train_data, self.train_data_adv = train
        else:
            self.train_data = train
        self.test_evaluator = test_evaluator
        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        if params.adversarial:
            if params.dataset.startswith('drugbank') :
                self.random_layer = RandomLayer([graph_classifier.cdan_dim, params.num_rels], 500).to(params.device)
            else:
                self.random_layer = RandomLayer([graph_classifier.cdan_dim, 2], 500).to(params.device)
            self.random_layer.device(params.device)
            self.ad_net = AdversarialNetwork(500, 500).to(params.device)
            if params.optimizer == "SGD":
                self.optimizer_ad = optim.SGD(self.ad_net.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
            if params.optimizer == "Adam":
                self.optimizer_ad = optim.Adam(self.ad_net.parameters(), lr=params.lr, weight_decay=self.params.l2)

        if not params.dataset.startswith('drugbank') :
            self.criterion = nn.BCELoss(reduce=False) 
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    # def validate(self):
    #     # self.load_model()
    #     if self.valid_evaluator and self.params.eval_every_iter:
    #         tic = time.time()
    #         if self.params.dataset.startswith('drugbank'):
    #             result, save_dev_data, _, _1 = self.valid_evaluator.eval()
    #             test_result, save_test_data, _, class_accuracies = self.test_evaluator.eval()
    #         else:
    #             result, save_dev_data = self.valid_evaluator.eval()
    #             test_result, save_test_data = self.test_evaluator.eval()

    #         logging.info('\033[95m Eval Performance:' + str(result) + ' in ' + str(time.time() - tic) + '\033[0m')
    #         logging.info('\033[93m Test Performance:' + str(test_result) + ' in ' + str(time.time() - tic) + '\033[0m')

    #         with open(self.params.save_result, "a") as f:
    #             time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    #             f.write(f'{time_now}:hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Eval Performance: {result}\n')
    #             f.write(f'{time_now}:hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Test Performance: {test_result}\n')

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        if self.params.adversarial:
            dataloader_adv = DataLoader(self.train_data_adv, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))
        for b_idx, batch in bar:
            #data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
            if self.params.adversarial:
                batch_adv = next(iter(dataloader_adv))
                data_pos_adv, r_labels_pos_adv, targets_pos_adv = self.params.move_batch_to_device(batch_adv, self.params.device)
                self.optimizer.zero_grad()
                self.optimizer_ad.zero_grad()
                score_pos, final_layer = self.graph_classifier(data_pos)
                score_pos_adv, final_layer_adv = self.graph_classifier(data_pos_adv)
                if not self.params.dataset.startswith('drugbank') :
                    m = nn.Sigmoid()
                    score_pos = m(score_pos)
                    score_pos_adv = m(score_pos_adv)
                    pred_pos = torch.flatten(torch.cat([1 - score_pos.unsqueeze(2), score_pos.unsqueeze(2)], dim=2).permute(1, 0, 2), start_dim=0, end_dim=1)
                    pred_pos_adv = torch.flatten(torch.cat([1 - score_pos_adv.unsqueeze(2), score_pos_adv.unsqueeze(2)], dim=2).permute(1, 0, 2), start_dim=0, end_dim=1)
                    pred_comb = torch.cat([pred_pos, pred_pos_adv], dim=0)
                    final_layer_comb = torch.cat([final_layer.repeat(209, 1), final_layer_adv.repeat(209, 1)], dim=0)
                    ad_loss = CDAN([final_layer_comb, pred_comb], self.ad_net, self.params.device, None, None, self.random_layer) * self.params.adversarial_weight
                    targets_pos = targets_pos.unsqueeze(1)
                    loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                    loss = torch.sum(loss_train * r_labels_pos)
                    loss += ad_loss
                else:
                    softmax_pred = nn.Softmax(dim=1)(score_pos)
                    softmax_pred_adv = nn.Softmax(dim=1)(score_pos_adv)
                    pred_comb = torch.cat([softmax_pred, softmax_pred_adv], dim=0)
                    final_layer_comb = torch.cat([final_layer, final_layer_adv], dim=0)
                    ad_loss = CDAN([final_layer_comb, pred_comb], self.ad_net, self.params.device, None, None, self.random_layer) * self.params.adversarial_weight 
                    loss = self.criterion(score_pos, r_labels_pos)
                    loss += ad_loss          
                loss.backward()
                clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.updates_counter += 1
                bar.set_description('epoch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()) + '| adversarial loss: ' + str(ad_loss.cpu().detach().numpy()))
            else:
                self.optimizer.zero_grad()
                score_pos = self.graph_classifier(data_pos)
                if not self.params.dataset.startswith('drugbank') :
                    m = nn.Sigmoid()
                    score_pos = m(score_pos)
                    targets_pos = targets_pos.unsqueeze(1)
                    loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                    loss = torch.sum(loss_train * r_labels_pos) 
                else:
                    loss = self.criterion(score_pos, r_labels_pos)           
                loss.backward()
                clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.updates_counter += 1
                bar.set_description('epoch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
    
            # except RuntimeError:
            #     print(data_pos, r_labels_pos, targets_pos)
            #    print('-------runtime error--------')
            #    continue
            with torch.no_grad():
                total_loss += loss.item()
                if self.params.dataset.startswith('drugbank') :
                    
                    label_ids = r_labels_pos.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                    #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
                    all_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                if self.params.dataset.startswith('drugbank') :
                    result, save_dev_data, _ = self.valid_evaluator.eval()
                    test_result, save_test_data, class_accuracies = self.test_evaluator.eval()
                else:
                    result, save_dev_data= self.valid_evaluator.eval()
                    test_result, save_test_data = self.test_evaluator.eval()
                logging.info('\033[95m Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'\033[0m')
                logging.info('\033[93m Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')

                with open(self.params.save_result, "a") as f:
                    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    f.write(f'{time_now}:hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Eval Performance: {result}\n')
                    f.write(f'{time_now}:hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Test Performance: {test_result}\n')
                
                # metric: acc auc:f1
                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.best_result_time = time_now
                    self.best_result = result
                    self.best_test_result = test_result
                    if self.params.dataset.startswith('drugbank') :
                        self.best_save_test_data = save_test_data
                        self.best_class_accuracies = class_accuracies
                    self.not_improved_count = 0
                    if self.params.dataset.startswith('drugbank') :
                        # logging.info('\033[93m Test Performance Per Class:' + str(save_test_data) + 'in ' + str(time.time() - tic)+'\033[0m')
                        pass
                    else:
                        with open('experiments/%s/result.json'%(self.params.experiment_name), 'a') as f:
                            f.write(json.dumps(save_test_data))
                            f.write('\n')
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))
        if self.params.dataset.startswith('drugbank'):
            auc = metrics.f1_score(all_labels, all_scores, average='macro')
            auc_pr = metrics.f1_score(all_labels, all_scores, average='micro')
        
            return total_loss/b_idx, auc, auc_pr, weight_norm
        else:
            return total_loss/b_idx, 0, 0, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            
            loss, auc, auc_pr, weight_norm = self.train_epoch()

            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier' + ('_adv' if self.params.adversarial else '') + '_chk.pth'))
        with open(self.params.save_result, "a") as f:
            f.write(f'best result: hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Eval Performance: {self.best_result} Test Performance: {self.best_test_result}\n')
        if self.params.dataset.startswith('drugbank') :
            with open(self.params.save_result, "a") as f:
                # f.write(f'hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Test Performance Per Relation Type: {self.best_save_test_data}\n')
                f.write(f'hop_{self.params.hop}_batch_{self.params.batch_size}_emb_{self.params.emb_dim}_b_{self.params.num_bases}_lr_{self.params.lr}_l2_{self.params.l2}_seed_{self.params.seed} Test Performance Per Class: {self.best_class_accuracies}\n')

    def case_study(self):
        self.reset_training_state()
        test_result, save_test_data = self.test_evaluator.print_result()

    def save_classifier(self):
        best_graph_classifier_path = f'best_graph_classifier_{self.params.setting}' + ('_adv' if self.params.adversarial else '') + '.pth'
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, best_graph_classifier_path))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
