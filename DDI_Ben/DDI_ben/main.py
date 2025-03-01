import os
import setproctitle
import argparse
from trainer import Trainer
from utils import *
import torch
import numpy as np

print('pid:', os.getpid())

def main():
    # set process name
    setproctitle.setproctitle('BNbench')

    # set hyperparameters
    parser = argparse.ArgumentParser(description='Task Aware Relation Graph for Few-shot Chemical Property Prediction')

    # general hyperparameters
    parser.add_argument('--model', type=str, default='TIGER', choices=['ComplEx', 'MSTE', 'MLP', 'Decagon', 'TIGER'])
    parser.add_argument('--problem', type=str, default='DDI', choices=['DDI'])
    parser.add_argument('--DDIsetting', type=str, default='all', choices=['S0', 'S1', 'S2', 'all'])
    parser.add_argument('--bionet', type=str, default='HetioNet', choices=['HetioNet', 'PrimeKG'])
    parser.add_argument('--name', default='testrun', help='Set run name for saving/restoring models')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='drugbank', choices=['drugbank', 'twosides'])
    parser.add_argument('--dataset_type', type=str, default='finger', choices=['random', 'finger', 'scaffold'])
    parser.add_argument('--gamma_split', type=str, default="55", choices=["55","60","65","70"])

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing') ### usually 0-1
    parser.add_argument("--epoch", type=int, default=200, help="training epoch")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--use_feat', default=1, type=bool, help='Whether to use drug feature')
    parser.add_argument('--use_reverse_edge', default=0, type=bool, help='Whether to add reverse edges in the training step')
    parser.add_argument('--data_aug', default=0, type=bool, help='Whether to add data as augmentation')

    parser.add_argument('--seed', default=100, type=int, help='Seed for randomization')
    parser.add_argument('--eval_skip', default=1, type=int, help='Evaluate every x epochs')
    parser.add_argument('--patience', default=20, type=int, help='Patience for early stopping')
    parser.add_argument('--adversarial', default=0, type=int, help='whether use adversarial training')
    parser.add_argument('--adversarial_task', default='S1', type=str, help='whether use adversarial training')
    parser.add_argument('--adversarial_weight', default=1, type=float, help='the weight of adversarial loss')
    parser.add_argument('--adversarial_mode', default=3, type=int, help='adversarial mode, 0 as the initial case, we have 3 other modes as recorded')

    args, remaining_args = parser.parse_known_args()
    
    # KGE models
    if args.model in ['ComplEx', 'MSTE']:
        parser.add_argument('--kge_dim', type=int, default=200, help='hidden dimension.')
        parser.add_argument('--kge_gamma', type=int, default=1, help='gamma parameter.')
        parser.add_argument('--kge_dropout', type=float, default=0, help='dropout rate.') ### DDI best 0
        parser.add_argument('--kge_loss', type=str, default='BCE_mean',  help='loss function')
    
    # MLP model
    elif args.model == 'MLP':
        parser.add_argument('--mlp_dropout', type=float, default=0.1, help='dropout rate.')
        parser.add_argument('--mlp_dim', type=int, default=200, help='hidden dimension.')

    # Decagon model decagon_drop
    elif args.model == 'Decagon':
        parser.add_argument('--decagon_dim', type=int, default=200, help='hidden dimension.')
        parser.add_argument('--decagon_drop', type=float,	default=0.1, help='Dropout to use in Decagon model')

    # set basic configurations
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model in ['ComplEx', 'MSTE']:
        args.use_feat = 0

    if args.dataset in ['twosides']:
        if args.model in ['ComplEx', 'MSTE']:
            args.batch_size = 512 # 2048

    if args.model == 'TIGER':
        args.patience = 10

    if args.data_aug:
        args.batch_size = 512

    args.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"

    # Training step in the trainer
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
