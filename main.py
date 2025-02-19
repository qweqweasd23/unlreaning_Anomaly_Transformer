import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    if config.mode == 'finetune':
        solver.finetune()
    elif config.mode == 'test':
        solver.test()

    return solver
def parse_target_columns(columns_str):
    return list(map(int, columns_str.split(',')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=float, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','finetune'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--target_columns', type=parse_target_columns, default='0', help='Target columns as a comma-separated list, need to -1 on actual col_num')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--masking_cols', type=parse_target_columns, default=None)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--scheduler_patience', type=int, default=1, help='learning rate scheduler')
    parser.add_argument('--prior_weight', type=int, default=1, help='valance beteween series_loss & prior_loss')
    parser.add_argument('--memmap_TF', type=int, default=0, help='np.load -> 0, np.memmap ->1')
    parser.add_argument('--total_row', type=int, default=None)
    parser.add_argument('--min_anomaly_length', type=int, default=1, help='min coutinuos anomaly sign, using in test_mode only')
    parser.add_argument('--soft_thres_ratio', type=float, default=0.0)
    parser.add_argument('--aggregate_unit', type=int, default=1)
    parser.add_argument('--zscore_threshold', type=int, default=2)
    
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
