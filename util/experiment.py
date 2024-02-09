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


class CustomScheduler:
    def __init__(self, initial_value, step_size, gamma):
        """
        Args:
            initial_value (float): Starting value of the variable.
            step_size (int): Period of variable update.
            gamma (float): Multiplicative factor of variable decay.
        """
        self.value = initial_value
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        """Update the variable based on the current epoch."""
        if self.epoch % self.step_size == 0:
            self.value *= self.gamma
        self.epoch += 1

    def get_value(self):
        """Return the current value of the variable."""
        return self.value


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-n', type=str, default='baseline',
                            help="The checkpoints and logs will be save in /experiments/$EXP_NAME")
    arg_parser.add_argument('--seeg_dir', '-sd', type=str, default='/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring', help="Root of the sEEG data")
    arg_parser.add_argument('--autoencoder_lr', '-al', type=float, default=1e-3, help="Autoencoder learning rate")
    arg_parser.add_argument('--classifier_lr', '-cl', type=float, default=1e-3, help="Classifier learning rate")
    arg_parser.add_argument('--autoencoder_epochs', '-ae', type=int, default=1000, help="Autoencoder training epochs")
    arg_parser.add_argument('--classifier_epochs', '-ce', type=int, default=100, help="Classifier training epochs")
    arg_parser.add_argument('--ckpt', '-c', type=str, default=None, help="Path to the checkpoint file")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=30, help="Batch size")
    arg_parser.add_argument('--train_ratio', '-r', type=float, default=0.7,
                            help="The ratio of training data. The rest is validation data")
    arg_parser.add_argument('--num_workers', '-w', type=int, default=4, help="Number of workers for dataloader")
    arg_parser.add_argument('--alpha', '-a', type=float, default=1.0, help="Weight of the contrastive loss")
    arg_parser.add_argument('--lr_step_size', '-ls', type=int, default=30, help="Step size for the learning rate scheduler")
    arg_parser.add_argument('--lr_gamma', '-lg', type=float, default=0.1, help="Gamma for the learning rate scheduler")
    arg_parser.add_argument('--alpha_step_size', '-as', type=int, default=200, help="Step size for the alpha scheduler")
    arg_parser.add_argument('--alpha_gamma', '-ag', type=float, default=10, help="Gamma for the alpha scheduler")
    args = arg_parser.parse_args()
    return args


