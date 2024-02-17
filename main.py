import os
import torch
import wandb
from util.experiment import set_seeds, get_args, print_num_params
from torch.utils.data import random_split, DataLoader
from model.classifier import E2eClassifier
from train.train import train_e2e_classifier
from eval.eval import val_e2e_classifier
from dataset.dataset4individual import Dataset4Individual
from util.experiment import CustomScheduler
from util.data import ID_2_IDX_CHANNEL

wandb.login(key="99528c40ebd16fca6632e963a943b99ac8a5f4b7")


def main(args):
    # Set up the experiment folder
    cwd = os.getcwd()
    exp_folder = os.path.join(cwd, 'experiments', args.exp_name)
    ckpt_folder = os.path.join(exp_folder, 'ckpt')
    log_folder = os.path.join(exp_folder, 'log')
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Set up the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the datasets and dataloaders
    print('Loading datasets and dataloaders ...')
    seeg_dir = args.seeg_dir
    ids = [key for key in sorted(ID_2_IDX_CHANNEL, key=lambda x: ID_2_IDX_CHANNEL[x][0])]
    phase = ['Encoding', 'SameDayRecall', 'NextDayRecall']
    train_loaders = []
    val_loaders = []
    for id in ids:
        dataset = Dataset4Individual(id, phase, seeg_dir)
        train_size = int(args.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        print(f'ID: {id}, Train size: {train_size}, Val size: {val_size}')
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # Initialize WandB
    wandb.init(project="prj_brainstorm", config=vars(args))

    # Define the e2e classifier
    num_electrodes = [ID_2_IDX_CHANNEL[key][1] for key in
                      sorted(ID_2_IDX_CHANNEL, key=lambda x: ID_2_IDX_CHANNEL[x][0])]
    e2e_classifier = E2eClassifier(num_electrodes).to(device)

    print_num_params(e2e_classifier)

    # Define the optimizers
    optimizer = torch.optim.AdamW(e2e_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define the lr scheduler
    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        lr_scheduler = None

    # Define the alpha scheduler
    if args.use_alpha_scheduler:
        alpha_scheduler = CustomScheduler(args.alpha, args.alpha_step_size, args.alpha_gamma)
    else:
        alpha_scheduler = None

    best_val_acc = None
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        e2e_classifier.load_state_dict(ckpt_state['classifier'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_acc = ckpt_state['best_val_acc']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Training
        train_e2e_classifier(epoch, e2e_classifier, optimizer, lr_scheduler, alpha_scheduler, train_loaders, device, args.alpha)

        # Validation
        val_alpha = alpha_scheduler.get_alpha() if alpha_scheduler else args.alpha
        val_acc = val_e2e_classifier(e2e_classifier, val_loaders, device, val_alpha)

        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f'New best e2e classifier found at epoch {best_epoch}')

            state = {
                'classifier': e2e_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }

            ckpt_file = os.path.join(ckpt_folder, f'best-e2e-classifier.pth')
            torch.save(state, ckpt_file)

        state = {
            'classifier': e2e_classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': val_acc,
            'best_epoch': best_epoch,
            'epoch': epoch + 1,
        }
        torch.save(state, os.path.join(ckpt_folder, f'latest-e2e-classifier.pth'))


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
