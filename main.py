import os
import torch
import wandb
from util.experiment import set_seeds, get_args
from torch.utils.data import random_split, DataLoader, Subset
from model.videoencoder import VideoEncoderVdFt, VideoEncoderDino
from model.seegencoder import SEEGEncoder, SEEGEncoderCls
from train.train import train
from eval.eval import eval
from dataset.dataset import DinoDataset, VideoMAEDataset

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

    # Define the datasets and dataloaders
    print('Loading datasets and dataloaders ...')
    seeg_file = args.seeg_file
    video_dir = args.video_dir
    time_window = args.time_window
    if 'dino' in video_dir:
        dataset = DinoDataset(seeg_file, video_dir, time_window, sample_rate=args.sample_rate)
    elif 'videomae' in video_dir:
        dataset = VideoMAEDataset(seeg_file, video_dir, time_window)
    else:
        raise ValueError("The video directory must contain either 'dino' or 'videomae'")
    num_train = int(len(dataset) * args.train_ratio)
    num_val = (len(dataset) - num_train) // 2
    # num_test = len(dataset) - num_train - num_val
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, num_train + num_val))
    test_indices = list(range(num_train + num_val, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the video encoder
    print('Creating video encoder ...')
    video_encoder_ver = args.video_encoder_version
    if video_encoder_ver == 'vdft':
        video_encoder = VideoEncoderVdFt().to(device)
    elif video_encoder_ver == 'dino':
        input_dim = time_window * 30
        video_encoder = VideoEncoderDino(input_dim).to(device)
    else:
        raise ValueError("The video encoder version must be either 'vdft' or 'dino'")

    # Define the seeg encoder
    print('Creating sEEG encoder ...')
    seeg_encoder_ver = args.seeg_encoder_version
    num_heads = args.num_heads
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward
    input_length = args.seeg_len
    num_input_channels = args.seeg_num_channels
    if seeg_encoder_ver == 'orig':
        seeg_encoder = SEEGEncoder(num_heads, num_encoder_layers, dim_feedforward, num_input_channels, input_length).to(device)
    elif seeg_encoder_ver == 'cls':
        c = args.seeg_encoder_cls_c
        seeg_encoder = SEEGEncoderCls(num_heads, num_encoder_layers, dim_feedforward, c, num_input_channels, input_length).to(device)
    else:
        raise ValueError("The sEEG encoder version must be either 'orig' or 'proj'")

    # Define the optimizer
    optimizer = torch.optim.Adam(list(video_encoder.parameters()) + list(seeg_encoder.parameters()), lr=args.lr)

    t = args.temperature

    best_val_acc1 = None
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        seeg_encoder.load_state_dict(ckpt_state['seeg_encoder'])
        video_encoder.load_state_dict(ckpt_state['video_encoder'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_acc1 = ckpt_state['best_val_acc1']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f'Epoch: {epoch + 1}')

        # Training
        train(video_encoder, seeg_encoder, optimizer, train_loader, device, t)

        # Validation
        loss, acc1, acc5 = eval(video_encoder, seeg_encoder, val_loader, device, 'val', t)

        if best_val_acc1 is None or acc1 > best_val_acc1:
            best_val_acc1 = acc1
            best_epoch = epoch + 1
            print(f'New best model found at epoch {best_epoch}')
            state = {
                'seeg_encoder': seeg_encoder.state_dict(),
                'video_encoder': video_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc1': best_val_acc1,
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }
            ckpt_file = os.path.join(ckpt_folder, f'best_ckpt.pth')
            torch.save(state, ckpt_file)

        state = {
            'seeg_encoder': seeg_encoder.state_dict(),
            'video_encoder': video_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc1': best_val_acc1,
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
    test_loss, test_acc1, test_acc5 = eval(video_encoder, seeg_encoder, test_loader, device, 'test', t)


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
