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


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"GPU Memory Allocated: {allocated / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved:  {reserved / 1024**2:.2f} MB")
        print(f"GPU Memory Allocated: {allocated / 1024 ** 3:.3f} GB")
        print(f"GPU Memory Reserved:  {reserved / 1024 ** 3:.3f} GB")
    else:
        print("CUDA is not available. Are you sure a GPU is attached and the right PyTorch version is installed?")


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-n', type=str, default='baseline',
                            help="The checkpoints and logs will be save in /experiments/$EXP_NAME")
    arg_parser.add_argument('--seeg_file', '-sf', type=str, default='/gpfs/data/tserre/Shared/Brainstorm_2024/all_seeg_data.npy', help="Path to the sEEG data file")
    arg_parser.add_argument('--video_dir', '-vd', type=str, default='/gpfs/data/tserre/Shared/Brainstorm_2024/greenbook_dinos', help="Path to the video embeddings directory")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-5, help="Learning rate")
    arg_parser.add_argument('--num_epochs', '-e', type=int, default=1000, help="total number of epochs for training")
    arg_parser.add_argument('--ckpt', '-c', type=str, default=None, help="path to the checkpoint file")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=10, help="batch size")
    arg_parser.add_argument('--train_ratio', '-tr', type=float, default=0.7,
                            help="the ratio of training data to all data. The rest is validation data")
    arg_parser.add_argument('--sample_rate', '-sr', type=int, default=1, help="the sample rate for the seeg data")
    arg_parser.add_argument('--time_window', '-tw', type=int, default=5, help="the time window for the data")
    arg_parser.add_argument('--seeg_encoder_version', '-sv', type=str, default='orig', help="the seeg version to use, can be 'orig', 'cls' or 'scene'")
    arg_parser.add_argument('--seeg_encoder_cls_c', '-sec', type=int, default=10, help="the c parameter for the cls sEEG encoder")
    arg_parser.add_argument('--seeg_num_channels', '-sc', type=int, default=84, help="the number of channels for the sEEG data")
    arg_parser.add_argument('--video_encoder_version', '-vv', type=str, default='dino', help="the video version to use, can be 'vdft', 'dino' or 'scene'")
    arg_parser.add_argument('--num_workers', '-w', type=int, default=4, help="Number of workers for dataloader")
    arg_parser.add_argument('--num_heads', '-nh', type=int, default=6, help="Number of heads for the sEEG encoder")
    arg_parser.add_argument('--num_encoder_layers', '-ne', type=int, default=6, help="number of encoder layers for the "
                                                                                     "sEEG encoder")
    arg_parser.add_argument('--dim_feedforward', '-df', type=int, default=1024, help="Hidden dimension for the sEEG "
                                                                                     "encoder")
    arg_parser.add_argument('--temperature', '-t', type=float, default=0.07, help="Temperature for the loss function")
    args = arg_parser.parse_args()
    return args


def print_num_params(model):
    """
    Get the number of parameters in a model.

    Parameters:
        model (`torch.nn.Module`): The model.

    Returns:
        num_params (`int`): The number of parameters in the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    return num_params


