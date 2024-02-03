import torch
from eval.eval import AverageMeter
from tqdm import tqdm
from util.loss import recon_loss, general_contrast_loss, agg_loss


def train(epoch, model, optimizer, train_loader, writer, device):
    model.train()

    # Initialize average meters
    recon_loss_meter = AverageMeter()
    contrast_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    # TODO: Utilize `video_idx` to save memory
    for seeg, video_idx, phase in tqdm(train_loader):
        batch_size = seeg.shape[0]

        seeg = seeg.to(device)
        video_idx = video_idx.to(device)

        optimizer.zero_grad()

        # Forward
        seeg_recon, embed = model(seeg)

        # Compute loss
        r_loss = recon_loss(seeg, seeg_recon)
        sim = embed @ embed.transpose(1, 0)
        c_loss = general_contrast_loss(sim, video_idx)
        total_loss = agg_loss(r_loss, c_loss)

        total_loss.backward()
        optimizer.step()

        # update metric
        with torch.no_grad():
            recon_loss_meter.update(r_loss.item(), batch_size)
            contrast_loss_meter.update(c_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), 1)

    writer.add_scalar(f'Train/Avg Reconstruction Loss of Each Epoch', recon_loss_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Contrastive Loss of Each Epoch', contrast_loss_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Total Loss of Each Epoch', total_loss_meter.avg, epoch + 1)
    print(f'Epoch: {epoch + 1}')
    print(f'Recontruction Loss: {recon_loss_meter.avg:.4f}')
    print(f'Contrastive Loss: {contrast_loss_meter.avg:.4f}')
    print(f'Total Loss: {total_loss_meter.avg:.4f}')
