import torch
import math
from util.experiment import AverageMeter, compute_top_k_acc
from tqdm import tqdm
import torch.nn.functional as F


def train(epoch, video_encoder, seeg_encoder, optimizer, train_loader, writer, device):
    # Set the model to train mode
    video_encoder.train()
    seeg_encoder.train()

    # Initialize average meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top2_acc_meter = AverageMeter()

    num_iter = len(train_loader) * epoch

    for video, seeg, seeg_padding_mask in tqdm(train_loader):
        batch_size = video.shape[0]

        video = video.to(device)
        seeg = seeg.to(device)
        seeg_padding_mask = seeg_padding_mask.to(device)

        optimizer.zero_grad()

        # Forward
        video_embedding = video_encoder(video)
        seeg_embedding = seeg_encoder(seeg, seeg_padding_mask)

        # Flatten the output for later similarity computation
        video_embedding = video_embedding.flatten(1, 2)
        seeg_embedding = seeg_embedding.flatten(1, 2)

        # Compute similarity
        sim = (video_embedding @ seeg_embedding.transpose(1, 0)) * math.e

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(sim, labels)

        # update metric
        loss_meter.update(loss.item(), batch_size)
        acc1, acc2 = compute_top_k_acc(sim, labels, topk=[1, 2])
        top1_acc_meter.update(acc1.item(), batch_size)
        top2_acc_meter.update(acc2.item(), batch_size)

        loss.backward()
        optimizer.step()

        num_iter += 1
        if num_iter % 50 == 0:
            writer.add_scalar('Train/Loss', loss_meter.avg, num_iter)
            writer.add_scalar('Train/Acc@1', top1_acc_meter.avg, num_iter)
            writer.add_scalar('Train/Acc@2', top2_acc_meter.avg, num_iter)

    print(f'Epoch: {epoch}')
    print(f'Train Acc@1 {top1_acc_meter.avg:.3f}')
    print(f'Train Acc@2 {top2_acc_meter.avg:.3f}')