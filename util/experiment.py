import os
import random
import argparse
import numpy as np
import torch


def set_seeds(seed=42):
    """
    Set random seeds to ensure that results can be reproduced.

    Parameters:
        seed (`int`): The random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, default='lr_1e-3-batch_10-train_ratio-0.8',
                            help="The checkpoints and logs will be save in /experiments/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-3, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=20, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true',
                            help="whether to load saved the latest checkpoint from $EXP_NAME and continue training")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=10, help="batch size")
    arg_parser.add_argument('--data_file', '-d', type=str, default='./data/data_segmented.npy',
                            help="path to the .npy file containing the data")
    arg_parser.add_argument('--train_ratio', '-r', type=float, default=0.7,
                            help="the ratio of training data to all data. 1/3 of the remaining data will be used for "
                                 "testing and 2/3 for validation")
    arg_parser.add_argument('--num_workers', '-w', type=int, default=4, help="Number of workers for dataloader")
    arg_parser.add_argument('--num_output_channels', '-o', type=int, default=64,
                            help="number of output channels for the seeg encoder")
    arg_parser.add_argument('--num_heads', '-nh', type=int, default=2, help="Number of heads for the sEEG encoder")
    arg_parser.add_argument('--num_encoder_layers', '-nl', type=int, default=6, help="number of encoder layers for the "
                                                                                     "sEEG encoder")
    arg_parser.add_argument('--dim_feedforward', '-f', type=int, default=2048, help="Hidden dimension for the seeg "
                                                                                    "encoder")
    args = arg_parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def compute_top_k_acc(pred, target, top_k=[1, ]):
    """
    Compute the top-k accuracy for the specified values of k.

    Parameters:
        pred (`torch.Tensor`): The predicted labels.
        target (`torch.Tensor`): The ground truth labels.
        top_k (`list`): The values of k for which the top-k accuracy will be computed.

    Returns:
        top_k_acc (`list`): The top-k accuracy for the specified values of k.
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, top_k_pred = pred.topk(max_k, 1)
        top_k_pred = top_k_pred.t()
        correct = top_k_pred.eq(target.view(1, -1).expand_as(top_k_pred))

        top_k_acc = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            top_k_acc.append(correct_k.mul_(100.0 / batch_size))
        return top_k_acc
