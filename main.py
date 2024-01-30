import os
import torch
from util.experiment import set_seeds, get_args
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from model.videoencoder import VideoEncoder
from model.seegencoder import SEEGEncoder
from train.train import train
from eval.eval import eval
from dataset.dataset4individual import Dataset4Individual


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

    # Set up the pretrained version of the video encoder and the video processor.
    ckpt = "MCG-NJU/videomae-base"

    # Define the datasets and dataloaders
    print('Loading datasets and dataloaders ...')
    id = 'e0010GP'
    phase = 'Encoding'
    seeg_dir = args.seeg_dir
    video_dir = args.video_dir
    num_frame_2_sample = 16
    dataset = Dataset4Individual(id, phase, seeg_dir, video_dir, ckpt, num_frame_2_sample)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f'Train size: {train_size}, Val size: {val_size}')
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the video encoder
    print('Creating video encoder ...')
    video_encoder = VideoEncoder(ckpt).to(device)

    # Define the seeg encoder
    print('Creating sEEG encoder ...')
    max_num_input_channels = 110
    num_output_channels = args.num_output_channels
    input_length = 5120
    output_length = 1568
    num_heads = args.num_heads
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward
    seeg_encoder = SEEGEncoder(max_num_input_channels, num_output_channels, input_length, output_length, num_heads,
                               num_encoder_layers, dim_feedforward).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(seeg_encoder.parameters(), lr=args.lr)

    best_val_acc1 = 0.0
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        seeg_encoder.load_state_dict(ckpt_state['seeg_encoder'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_acc1 = ckpt_state['best_val_acc1']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        # Training
        train(epoch, video_encoder, seeg_encoder, optimizer, train_loader, writer, device)

        # Validation
        acc1, acc2 = eval(epoch, video_encoder, seeg_encoder, val_loader, writer, device, 'val')

        if acc1 > best_val_acc1:
            best_val_acc1 = acc1
            best_epoch = epoch + 1
            print(f'New best model found at epoch {best_epoch}')

        state = {
            'seeg_encoder': seeg_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc1': best_val_acc1,
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
