import os
import torch
import wandb
from util.experiment import set_seeds, get_args
from torch.utils.data import random_split, DataLoader
from model.videoencoder import VideoEncoder
from model.seegencoder import SEEGEncoder
from train.train import train
from eval.eval import eval
from dataset.dataset import CustomDataset

wandb.login(key="99528c40ebd16fca6632e963a943b99ac8a5f4b7")


def main(args):
    # Set up the experiment folder
    exp_folder = os.path.join('./experiments', args.exp_name)
    ckpt_folder = os.path.join(exp_folder, 'ckpt')
    log_folder = os.path.join(exp_folder, 'log')
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Initialize WandB
    wandb.init(project="prj_brainstorm", config=vars(args))

    # Set up the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the pretrained version of the video encoder and the video processor.

    # Define the datasets and dataloaders
    print('Loading datasets and dataloaders ...')
    seeg_file = args.seeg_file
    video_dir = args.video_dir
    dataset = CustomDataset(seeg_file, video_dir)
    num_train = int(len(dataset) * args.train_ratio)
    num_val = (len(dataset) - num_train) // 2
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the video encoder
    print('Creating video encoder ...')
    ckpt = "MCG-NJU/videomae-base"
    video_encoder = VideoEncoder(ckpt).to(device)

    # Define the seeg encoder
    print('Creating sEEG encoder ...')
    num_heads = args.num_heads
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward
    seeg_encoder = SEEGEncoder(num_heads, num_encoder_layers, dim_feedforward).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(seeg_encoder.parameters(), lr=args.lr)

    best_val_loss = None
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        seeg_encoder.load_state_dict(ckpt_state['seeg_encoder'])
        video_encoder.load_state_dict(ckpt_state['video_encoder'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_acc1 = ckpt_state['best_val_loss']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f'Epoch: {epoch + 1}')

        # Training
        train(video_encoder, seeg_encoder, optimizer, train_loader, device)

        # Validation
        loss, acc1, acc2 = eval(video_encoder, seeg_encoder, val_loader, device, 'val')

        if best_val_loss is None or loss < best_val_loss:
            best_val_loss = loss
            best_epoch = epoch + 1
            print(f'New best model found at epoch {best_epoch}')
            state = {
                'seeg_encoder': seeg_encoder.state_dict(),
                'video_encoder': video_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }
            ckpt_file = os.path.join(ckpt_folder, f'best_ckpt.pth')
            torch.save(state, ckpt_file)

        state = {
            'seeg_encoder': seeg_encoder.state_dict(),
            'video_encoder': video_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epoch': epoch + 1,
        }
        ckpt_file = os.path.join(ckpt_folder, f'latest_ckpt.pth')
        torch.save(state, ckpt_file)

    # Testing
    print('Testing ...')
    ckpt_file = os.path.join(ckpt_folder, f'best_ckpt.pth')
    ckpt_state = torch.load(ckpt_file)
    seeg_encoder.load_state_dict(ckpt_state['seeg_encoder'])
    video_encoder.load_state_dict(ckpt_state['video_encoder'])
    test_loss, test_acc1, test_acc2 = eval(video_encoder, seeg_encoder, test_loader, device, 'test')


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
