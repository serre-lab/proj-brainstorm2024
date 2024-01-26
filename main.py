import os
import torch
from util.experiment import set_seeds, get_args
from torch.utils.tensorboard import SummaryWriter
from model.videoencoder import VideoEncoder
from model.seegencoder import SEEGEncoder
from train.train import train
from eval.eval import eval


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
    # TODO: Define the datasets and dataloaders.
    print('Loading datasets and dataloaders ...')
    train_dataset = None
    val_dataset = None
    test_dataset = None
    train_loader = None
    val_loader = None
    test_loader = None

    # Define the video encoder
    print('Creating video encoder ...')
    ckpt = "MCG-NJU/videomae-base"
    video_encoder = VideoEncoder(ckpt)

    # Define the seeg encoder
    print('Creating sEEG encoder ...')
    # TODO: Use the correct max_num_input_channels.
    max_num_input_channels = 100
    num_output_channels = args.num_output_channels
    # TODO: Use the correct input_length
    input_length = 2560
    output_length = 1568
    num_heads = args.num_heads
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward
    seeg_encoder = SEEGEncoder(max_num_input_channels, num_output_channels, input_length, output_length, num_heads,
                               num_encoder_layers, dim_feedforward)

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
        acc1, _ = eval(epoch, video_encoder, seeg_encoder, val_loader, writer, device, 'val')

        if acc1 > best_val_acc1:
            best_val_acc1 = acc1
            best_epoch = epoch + 1

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

    # Testing
    if best_epoch <= start_epoch:
        print('No best model found. Skip testing.')
    else:
        print(f'Testing the best model at epoch {best_epoch}...')
        seeg_encoder.load_state_dict(torch.load(os.path.join(ckpt_folder, f'epoch_{best_epoch}.pth'))['seeg_encoder'])
        eval(None, video_encoder, seeg_encoder, test_loader, writer, device, 'test')


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
