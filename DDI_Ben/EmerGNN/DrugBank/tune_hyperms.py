import os
import argparse
import torch
import random
from load_data import DataLoader

from base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser for EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
parser.add_argument('--dataset', type=str, default='S1_1', help='the dataset to use')
parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to load.')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size')
parser.add_argument('--seed', type=int, default=1234)

class options:
    def __init__():
        pass

if __name__ == '__main__':
    args = parser.parse_args()
    args.relation_class = {'Pharmacokinetic interactions - Absorption interactions': [2, 12, 17, 61, 66],
                'Pharmacokinetic interactions - Distribution interacitons': [42, 44, 72, 74],
                'Pharmacokinetic interactions - Metabolic interactions': [3, 10, 46],
                'Pharmacokinetic interactions - Excretion interactions': [64, 71],
                'Pharmacodynamic interactions - Additive or synergistic effects': [0, 1, 5, 6, 7, 8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 
                24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 45, 51, 52, 53, 54, 55, 56, 58, 59, 62, 63, 67, 68, 70,
                73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                'Pharmacodynamic interacitons - Antagonistic effects': [4, 11, 13, 16, 25, 28, 36, 47, 48, 49, 50, 57, 60, 65, 69, 75]}
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, eval_rel
    KG = dataloader.KG
    vKG = dataloader.vKG
    tKG = dataloader.tKG
    triplets = dataloader.triplets
    train_pos, train_neg = torch.LongTensor(triplets['train']).cuda(), None
    valid_pos, valid_neg = torch.LongTensor(triplets['valid']).cuda(), None
    test_pos,  test_neg  = torch.LongTensor(triplets['test']).cuda(), None

    if not os.path.exists('results'):
        os.makedirs('results')

    def run_model(params):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.lr = params['lr']
        args.lamb = params['lamb']
        args.length = params['length']
        args.n_batch = params['n_batch']
        args.n_dim = params['n_dim']
        args.feat = params['feat']

        model = BaseModel(eval_ent, eval_rel, args)
        best_acc = -1
        early_stop = 0
        try:
            for e in range(args.n_epoch):
                if early_stop > 3:
                    break
                dataloader.shuffle_train()
                KG = dataloader.KG
                train_pos = torch.LongTensor(dataloader.train_data).cuda()
                model.train(train_pos, None, KG)
                if (e+1) % args.epoch_per_test == 0:
                    v_f1, v_acc, v_kap, v_acc_six_class, v_acc_per_class = model.evaluate(valid_pos, valid_neg, vKG)
                    t_f1, t_acc, t_kap, t_acc_six_class, t_acc_per_class = model.evaluate(test_pos, test_neg, tKG)
                    out_str = 'epoch:%d\tfeat:%s lr:%.6f lamb:%.8f n_batch:%d n_dim:%d layer:%d\t[Valid] f1:%.4f acc:%.4f kap:%.4f\t[Test] f1:%.4f acc:%.4f kap:%.4f' % (e+1, args.feat, args.lr, args.lamb, args.n_batch, args.n_dim, args.length, v_f1, v_acc, v_kap, t_f1, t_acc, t_kap)
                    if v_f1 > best_acc:
                        best_acc = v_f1
                        best_str = out_str
                        early_stop = 0
                    else:
                        early_stop += 1
        except RuntimeError as e:
            print(e)
            return 0

        print(best_str)
        with open(os.path.join('results', args.dataset+'_tune.txt'), 'a+') as f:
            f.write(best_str+'\n\n')
        return -best_acc

    space = {
        "lr": hp.choice("lr", [3e-3, 1e-3, 3e-4]),
        "lamb": hp.choice("lamb", [1e-8, 1e-6, 1e-4, 1e-2]),
        "n_batch": hp.choice("n_batch", [32, 64]),
        "n_dim": hp.choice("n_dim", [32, 64]),
        "length": hp.choice("length",[2, 3]),
        "feat": hp.choice("feat", ['M', 'E']),
    } # space has been changed

    trials = Trials()
    best = fmin(run_model, space, algo=partial(tpe.suggest, n_startup_jobs=60), max_evals=30, trials=trials)
    print(best)
    print("--------------------------------end---------------------------------")
                

    

