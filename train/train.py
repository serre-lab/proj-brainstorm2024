import torch
from eval.eval import AverageMeter
from tqdm import tqdm
from util.loss import recon_loss, general_contrast_loss, agg_loss
import wandb


def train(epoch, model, optimizer, scheduler, train_loader, writer, device, alpha_value):
    model.train()

    # Initialize average meters
    recon_loss_meter = AverageMeter()
    contrast_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for seeg, video_idx, phase in tqdm(train_loader):
        batch_size = seeg.shape[0]

        seeg = seeg.to(device)
        video_idx = video_idx.to(device)

        optimizer.zero_grad()

        # Forward
        seeg_recon, embed_before = model(seeg)
        embed = embed_before.flatten(start_dim=1) 
        # Compute loss
        r_loss = recon_loss(seeg, seeg_recon)
        c_loss = general_contrast_loss(embed, video_idx)
        total_loss = agg_loss(r_loss, c_loss, alpha_value)

        total_loss.backward()
        optimizer.step()

        # update metric
        with torch.no_grad():
            recon_loss_meter.update(r_loss.item(), batch_size)
            contrast_loss_meter.update(c_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            
        wandb.log({"training_loss": total_loss_meter.avg,
                   "train_reconstruction_loss": recon_loss_meter.avg,
                   "train_contrastive_loss": contrast_loss_meter.avg,
                   "train_scaled_contrastive_loss": contrast_loss_meter.avg * alpha_value})

    scheduler.step()

    writer.add_scalar(f'Train/Avg Reconstruction Loss of Each Epoch', recon_loss_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Contrastive Loss of Each Epoch', contrast_loss_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Scaled Contrastive Loss of Each Epoch', (contrast_loss_meter.avg*alpha_value), epoch + 1)
    writer.add_scalar(f'Train/Avg Total Loss of Each Epoch', total_loss_meter.avg, epoch + 1)
    print(f'Epoch: {epoch + 1}')
    print(f'Recontruction Loss: {recon_loss_meter.avg:.4f}')
    print(f'Scaled Contrastive Loss: {contrast_loss_meter.avg*alpha_value:.4f}')
    print(f'Total Loss: {total_loss_meter.avg:.4f}')
