import torch
from eval.eval import AverageMeter
from tqdm import tqdm
from util.loss import recon_loss, general_contrast_loss, agg_loss
import wandb


def train_autoencoder(epoch, autoencoder, autoencoder_optimizer, lr_scheduler, alpha_scheduler, train_loader, device, alpha_value):
    autoencoder.train()

    # Initialize average meters
    recon_loss_meter = AverageMeter()
    contrast_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for seeg, video_idx, phase in tqdm(train_loader):
        batch_size = seeg.shape[0]

        seeg = seeg.to(device)
        video_idx = video_idx.to(device)

        autoencoder_optimizer.zero_grad()

        # Forward
        seeg_recon, embed_before = autoencoder(seeg)
        embed = embed_before.flatten(start_dim=1) 
        # Compute loss
        r_loss = recon_loss(seeg, seeg_recon)
        c_loss = general_contrast_loss(embed, video_idx)
        total_loss = agg_loss(r_loss, c_loss, alpha_value)

        total_loss.backward()
        autoencoder_optimizer.step()

        # update metric
        with torch.no_grad():
            recon_loss_meter.update(r_loss.item(), batch_size)
            contrast_loss_meter.update(c_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            
            wandb.log({"autoencoder_train_loss": total_loss_meter.avg,
                       "autoencoder_train_recon_loss": recon_loss_meter.avg,
                       "autoencoder_train_contra_loss": contrast_loss_meter.avg,
                       "autoencoder_train_scaled_contra_loss": contrast_loss_meter.avg * alpha_value})

    lr_scheduler.step()
    alpha_scheduler.step()

    print(f'Epoch: {epoch + 1}')
    print(f'Recontruction Loss: {recon_loss_meter.avg:.4f}')
    print(f'Scaled Contrastive Loss: {contrast_loss_meter.avg*alpha_value:.4f}')
    print(f'Total Loss: {total_loss_meter.avg:.4f}')


def train_classifier(epoch, autoencoder, classifier, classifier_optimizer, train_loader, device):
    autoencoder.eval()
    classifier.train()

    # Initialize average meters
    loss_meter = AverageMeter()

    for seeg, video_idx, phase in tqdm(train_loader):
        batch_size = seeg.shape[0]

        seeg = seeg.to(device)
        video_idx = video_idx.to(device)

        classifier_optimizer.zero_grad()

        with torch.no_grad():
            _, embed = autoencoder(seeg)

        output = classifier(embed)

        loss = torch.nn.CrossEntropyLoss()(output, video_idx)

        loss.backward()
        classifier_optimizer.step()

        with torch.no_grad():
            loss_meter.update(loss.item(), batch_size)
            wandb.log({"classifier_train_loss": loss_meter.avg})

    print(f'Epoch: {epoch + 1}')
    print(f'Loss: {loss_meter.avg:.4f}')
