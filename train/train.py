import torch
import math
import wandb
import torch.nn.functional as F
from eval.eval import AverageMeter, compute_top_k_acc
from tqdm import tqdm


def train(video_encoder, seeg_encoder, optimizer, train_loader, device):
    video_encoder.eval()
    seeg_encoder.train()

    # Initialize average meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top2_acc_meter = AverageMeter()

    for video, seeg in tqdm(train_loader):
        batch_size = video.shape[0]

        video = video.to(device)
        seeg = seeg.to(device)

        optimizer.zero_grad()

        # Forward
        video_embedding = video_encoder(video)
        seeg_embedding = seeg_encoder(seeg)

        # Flatten video and seeg embeddings
        video_embedding = video_embedding.view(batch_size, -1)
        seeg_embedding = seeg_embedding.view(batch_size, -1)

        # Compute similarity
        sim = (video_embedding @ seeg_embedding.transpose(1, 0)) * math.e

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(sim, labels)

        loss.backward()
        optimizer.step()

        # update metric
        with torch.no_grad():
            loss_meter.update(loss.item(), batch_size)
            acc1, acc2 = compute_top_k_acc(sim, labels, top_k=[1, 2])
            top1_acc_meter.update(acc1, batch_size)
            top2_acc_meter.update(acc2, batch_size)
            wandb.log({'Train/Loss': loss_meter.avg,
                       'Train/Acc@1': top1_acc_meter.avg,
                       'Train/Acc@2': top2_acc_meter.avg})

    print(f'Train/Loss {loss_meter.avg:.4f}')
    print(f'Train/Acc@1 {top1_acc_meter.avg:.4f}%')
    print(f'Train/Acc@2 {top2_acc_meter.avg:.4f}%')
