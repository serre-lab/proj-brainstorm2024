import os
import torch
import wandb
from util.experiment import set_seeds, get_args
from torch.utils.data import random_split, DataLoader
from model.autoencoder import ConvAutoEncoder
from model.classifier import LinearClassifier
from train.train import train_autoencoder, train_classifier
from eval.eval import val_autoencoder, val_classifier
from dataset.dataset4individual import Dataset4Individual
from util.experiment import CustomScheduler

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
    wandb.init(project="prj_brainstorm", config=vars(args))

    # Define the autoencoder and classifier
    autoencoder = ConvAutoEncoder().to(device)
    classifier = LinearClassifier().to(device)

    # Define the optimizers
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.autoencoder_lr)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr)

    # Define the lr scheduler
    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(autoencoder_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        lr_scheduler = None

    # Define the alpha scheduler
    if args.use_alpha_scheduler:
        alpha_scheduler = CustomScheduler(args.alpha, args.alpha_step_size, args.alpha_gamma)
    else:
        alpha_scheduler = None

    best_val_loss = None
    best_epoch = 0
    start_epoch = 0

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt_state = torch.load(args.ckpt)
        autoencoder.load_state_dict(ckpt_state['autoencoder'])
        autoencoder_optimizer.load_state_dict(ckpt_state['optimizer'])
        best_val_loss = ckpt_state['best_val_loss']
        best_epoch = ckpt_state['best_epoch']
        start_epoch = ckpt_state['epoch']

    for epoch in range(start_epoch, start_epoch + args.autoencoder_epochs):
        # Training
        train_autoencoder(epoch, autoencoder, autoencoder_optimizer, lr_scheduler, alpha_scheduler, train_loader, device, args.alpha)

        # Validation
        recon_loss, contrast_loss, total_loss = val_autoencoder(autoencoder, val_loader, device, args.alpha)

        if best_val_loss is None or total_loss < best_val_loss:
            best_val_loss = total_loss
            best_epoch = epoch + 1
            print(f'New best autoencoder found at epoch {best_epoch}')

            state = {
                'autoencoder': autoencoder.state_dict(),
                'optimizer': autoencoder_optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }

            ckpt_file = os.path.join(ckpt_folder, f'best-autoencoder.pth')
            torch.save(state, ckpt_file)

    # Train the classifier and test it on the validation set
    best_val_acc = None
    autoencoder.load_state_dict(torch.load(ckpt_file)['autoencoder'])
    for epoch in range(args.classifier_epochs):
        train_classifier(epoch, autoencoder, classifier, classifier_optimizer, train_loader, device)

        acc = val_classifier(autoencoder, classifier, val_loader, device)
        if best_val_acc is None or acc > best_val_acc:
            best_val_acc = acc
            print(f'New best classifier found at epoch {epoch + 1} with accuracy {best_val_acc}')
            ckpt_file = os.path.join(ckpt_folder, f'best-classifier.pth')
            torch.save(state, ckpt_file)


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
