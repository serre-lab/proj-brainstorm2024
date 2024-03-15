import torch
import wandb
import numpy as np
import torch.nn.functional as F
from eval.eval import AverageMeter, compute_top_k_acc
from tqdm import tqdm


def train(video_encoder, seeg_encoder, optimizer, train_loader, device, t, use_mask):
    video_encoder.train()
    seeg_encoder.train()

    # Initialize average meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()

    for data in tqdm(train_loader):
        if use_mask:
            video, video_mask, seeg, seeg_mask = data
        else:
            video, seeg = data

        batch_size = video.shape[0]

        video = video.to(device)
        seeg = seeg.to(device)
        video_mask = video_mask.to(device) if video_mask is not None else None
        seeg_mask = seeg_mask.to(device) if seeg_mask is not None else None

        optimizer.zero_grad()

        # Forward
        if use_mask:
            video = video_encoder(video, video_mask)
            seeg = seeg_encoder(seeg, seeg_mask)
        else:
            video = video_encoder(video)
            seeg = seeg_encoder(seeg)

        # Flatten video and seeg embeddings
        if len(video.shape) > 2:
            video = video.view(batch_size, -1)
        if len(seeg.shape) > 2:
            seeg = seeg.view(batch_size, -1)

        # Normalize embeddings
        video = F.normalize(video, p=2, dim=1)
        seeg = F.normalize(seeg, p=2, dim=1)

        # Compute similarity
        sim = (video @ seeg.transpose(1, 0)) * np.exp(t)

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss_1 = F.cross_entropy(sim, labels)
        loss_2 = F.cross_entropy(sim.transpose(1, 0), labels)
        loss = (loss_1 + loss_2) / 2

        loss.backward()
        optimizer.step()

        # update metric
        with torch.no_grad():
            loss_meter.update(loss.item(), batch_size)
            acc1, acc5 = compute_top_k_acc(sim, labels, top_k=[1, 5])
            top1_acc_meter.update(acc1, batch_size)
            top5_acc_meter.update(acc5, batch_size)

    with torch.no_grad():
        wandb.log({'Train/Loss': loss_meter.avg,
                   'Train/Acc@1': top1_acc_meter.avg,
                   'Train/Acc@5': top5_acc_meter.avg})
    print(f'Train/Loss {loss_meter.avg:.4f}')
    print(f'Train/Acc@1 {top1_acc_meter.avg:.4f}%')
    print(f'Train/Acc@5 {top5_acc_meter.avg:.4f}%')
