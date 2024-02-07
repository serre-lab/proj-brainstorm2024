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
    arg_parser.add_argument('--exp_name', '-n', type=str, default='baseline',
                            help="The checkpoints and logs will be save in /experiments/$EXP_NAME")
    arg_parser.add_argument('--seeg_dir', '-sd', type=str, default='/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring', help="Root of the sEEG data")
    arg_parser.add_argument('--video_dir', '-vd', type=str, default='./data/dev/Movie Clips/',  help="Root of the video"
                                                                                                     " data")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-3, help="Learning rate")
    arg_parser.add_argument('--num_epochs', '-e', type=int, default=2, help="total number of epochs for training")
    arg_parser.add_argument('--ckpt', '-c', type=str, default=None, help="path to the checkpoint file")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=30, help="batch size")
    arg_parser.add_argument('--train_ratio', '-tr', type=float, default=0.7,
                            help="the ratio of training data to all data. The rest is validation data")
    arg_parser.add_argument('--num_workers', '-w', type=int, default=4, help="Number of workers for dataloader")
    arg_parser.add_argument('--num_output_channels', '-no', type=int, default=768,
                            help="number of output channels for the seeg encoder")
    arg_parser.add_argument('--num_heads', '-nh', type=int, default=2, help="Number of heads for the sEEG encoder")
    arg_parser.add_argument('--num_encoder_layers', '-ne', type=int, default=6, help="number of encoder layers for the "
                                                                                     "sEEG encoder")
    arg_parser.add_argument('--dim_feedforward', '-df', type=int, default=2048, help="Hidden dimension for the sEEG "
                                                                                     "encoder")
    arg_parser.add_argument('--alpha', '-a', type=float, default=1.0, help="Hyperparameter to control the weight of the contrastive loss")
    args = arg_parser.parse_args()
    return args


