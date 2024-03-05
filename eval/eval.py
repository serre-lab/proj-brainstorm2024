import torch
import numpy as np
import wandb
import torch.nn.functional as F
from tqdm import tqdm


def eval(video_encoder, seeg_encoder, eval_loader, device, split, t):
    video_encoder.eval()
    seeg_encoder.eval()

    video_embeddings = None
    seeg_embeddings = None

    with torch.no_grad():
        for video, seeg in tqdm(eval_loader):
            video = video.to(device)
            seeg = seeg.to(device)

            # Forward
            video_embedding = video_encoder(video)
            seeg_embedding = seeg_encoder(seeg)

            if video_embeddings is None:
                video_embeddings = video_embedding
                seeg_embeddings = seeg_embedding
            else:
                video_embeddings = torch.cat((video_embeddings, video_embedding), dim=0)
                seeg_embeddings = torch.cat((seeg_embeddings, seeg_embedding), dim=0)

        # Flatten video and seeg embeddings
        # print gpu memory in use (in GB) for debugging
        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

        if len(video_embedding.shape) > 2:
            video_embeddings = video_embeddings.view(video_embeddings.shape[0], -1)
            seeg_embeddings = seeg_embeddings.reshape(seeg_embeddings.shape[0], -1)

        # Normalize embeddings
        video_embeddings = F.normalize(video_embeddings, p=2, dim=1)
        seeg_embeddings = F.normalize(seeg_embeddings, p=2, dim=1)

        # Compute similarity
        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

        sim = (video_embeddings @ seeg_embeddings.transpose(1, 0)) * np.exp(t)
        labels = torch.arange(video_embeddings.shape[0]).to(device)

        # Compute accuracy
        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.transpose(1, 0), labels)) / 2
        acc1, acc5 = compute_top_k_acc(sim, labels, top_k=[1, 5])

        wandb.log({f'{split}/Loss': loss,
                   f'{split}/Acc@1': acc1,
                   f'{split}/Acc@5': acc5})
        print(f'{split}/Loss {loss:.4f}')
        print(f'{split}/Acc@1 {acc1:.4f}%')
        print(f'{split}/Acc@5 {acc5:.4f}%')
        return loss, acc1, acc5


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_top_k_acc(pred, target, top_k=[1, ]):
    """
    Compute the top-k accuracy for the specified values of k.

    Parameters:
        pred (`torch.Tensor`): The predicted similarity matrix.
        target (`torch.Tensor`): The ground truth labels.
        top_k (`list`): The values of k for which the top-k accuracy will be computed.

    Returns:
        top_k_acc (`list`): The top-k accuracy for the specified values of k.
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, top_k_pred = pred.topk(max_k, 1)
        top_k_pred = top_k_pred.t()
        correct = top_k_pred.eq(target.view(1, -1).expand_as(top_k_pred))

        top_k_acc = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            top_k_acc.append(correct_k.mul_(100.0 / batch_size).item())
        return top_k_acc
