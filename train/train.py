import torch
import wandb
import numpy as np
import torch.nn.functional as F
from eval.eval import AverageMeter, compute_top_k_acc
from tqdm import tqdm

def train(video_encoder, seeg_encoder, optimizer, train_loader, device, t):
    video_encoder.train()
    seeg_encoder.train()

    # Initialize average meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()

    print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    for video_embedding, seeg_embedding in tqdm(train_loader):
        batch_size = video_embedding.shape[0]

        video_embedding = video_embedding.to(device)
        seeg_embedding = seeg_embedding.to(device)

        optimizer.zero_grad()
        # Forward
        video_embedding = video_encoder(video_embedding)
        seeg_embedding = seeg_encoder(seeg_embedding)

        # Flatten video and seeg embeddings
        if len(video_embedding.shape) > 2:
            video_embedding = video_embedding.view(batch_size, -1)
            seeg_embedding = seeg_embedding.reshape(batch_size, -1)

        # Normalize embeddings
        video_embedding = F.normalize(video_embedding, p=2, dim=1)
        seeg_embedding = F.normalize(seeg_embedding, p=2, dim=1)

        # Compute similarity
        sim = (video_embedding @ seeg_embedding.transpose(1, 0)) * np.exp(t)

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss_1 = F.cross_entropy(sim, labels)
        loss_2 = F.cross_entropy(sim.transpose(1, 0), labels)
        loss = (loss_1 + loss_2) / 2

        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

        loss.backward()

        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

        optimizer.step()

        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

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
