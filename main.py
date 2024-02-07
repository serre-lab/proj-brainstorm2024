import os
import torch
from util.experiment import set_seeds, get_args
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from model.autoencoder import AutoEncoder
from train.train import train
from eval.eval import eval
from dataset.dataset4individual import Dataset4Individual
import wandb

wandb.login(key="99528c40ebd16fca6632e963a943b99ac8a5f4b7")

def main(args):
    # Set up the experiment folder
    exp_folder = os.path.join('./experiments', args.exp_name)
    ckpt_folder = os.path.join(exp_folder, 'ckpt')
    log_folder = os.path.join(exp_folder, 'log')
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Set up the logger.
    writer = SummaryWriter(log_dir=log_folder)

    # Set up the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the datasets and dataloaders
    print('Loading datasets and dataloaders ...')
    id = 'e0017MC'
    phase = ['Encoding', 'SameDayRecall', 'NextDayRecall']
    seeg_dir = args.seeg_dir
    dataset = Dataset4Individual(id, phase, seeg_dir)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f'Train size: {train_size}, Val size: {val_size}')
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize WandB
    wandb.init(project="prj_brainstorm", config={"learning_rate":args.lr, "epochs":args.num_epochs, "alpha":args.alpha, "batch_size":args.batch_size})
    
    # Define the seeg encoder
    print('Creating sEEG encoder ...')
    model = AutoEncoder().to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = None
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        model.load_state_dict(ckpt_state['seeg_encoder'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_loss = ckpt_state['best_val_loss']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        # Training
        train(epoch, model, optimizer, train_loader, writer, device, args.alpha)

        # Validation
        recon_loss, contrast_loss, total_loss = eval(epoch, model, val_loader, writer, device, 'val')

        if best_val_loss is None:
            best_val_loss = total_loss
            best_epoch = epoch + 1
            print(f'New best model found at epoch {best_epoch}')
        elif total_loss < best_val_loss:
            best_val_loss = total_loss
            best_epoch = epoch + 1
            print(f'New best model found at epoch {best_epoch}')

        state = {
            'seeg_encoder': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epoch': epoch + 1,
        }
        ckpt_file = os.path.join(ckpt_folder, f'epoch_{epoch + 1}.pth')
        torch.save(state, ckpt_file)

    writer.close()


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
