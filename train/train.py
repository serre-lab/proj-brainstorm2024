import torch
import math
from eval.eval import AverageMeter, compute_top_k_acc
from tqdm import tqdm
import torch.nn.functional as F


def train(epoch, video_encoder, seeg_encoder, optimizer, train_loader, writer, device):
    # Set the model to train mode
    video_encoder.eval()
    seeg_encoder.train()

    # Initialize average meters
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top2_acc_meter = AverageMeter()

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
        acc1, acc2 = compute_top_k_acc(sim, labels, top_k=[1, 2])
        top1_acc_meter.update(acc1, batch_size)
        top2_acc_meter.update(acc2, batch_size)

        loss.backward()
        optimizer.step()

    writer.add_scalar(f'Train/Avg Loss for Each Epoch', loss_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Acc@1 for Each Epoch', top1_acc_meter.avg, epoch + 1)
    writer.add_scalar(f'Train/Avg Acc@2 for Each Epoch', top2_acc_meter.avg, epoch + 1)
    print(f'Epoch: {epoch + 1}')
    print(f'Average Train Loss {loss_meter.avg:.4f}')
    print(f'Average Train Acc@1 {top1_acc_meter.avg:.3f}')
    print(f'Average Train Acc@2 {top2_acc_meter.avg:.3f}')
