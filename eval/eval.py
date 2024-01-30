import torch
import math
from tqdm import tqdm


def eval(epoch, video_encoder, seeg_encoder, eval_loader, writer, device, split):
    video_encoder.eval()
    seeg_encoder.eval()

    video_embeddings = None
    seeg_embeddings = None

    with torch.no_grad():
        # TODO: Utilize `video_idx` to save memory
        for seeg, seeg_padding_mask, video, _ in tqdm(eval_loader):
            video = video.to(device)
            seeg = seeg.to(device)
            seeg_padding_mask = seeg_padding_mask.to(device)

            # Forward
            video_embedding = video_encoder(video)
            seeg_embedding = seeg_encoder(seeg, seeg_padding_mask)

            # Flatten the output for later similarity computation
            video_embedding = video_embedding.flatten(1, 2)
            seeg_embedding = seeg_embedding.flatten(1, 2)

            if video_embeddings is None:
                video_embeddings = video_embedding
                seeg_embeddings = seeg_embedding
            else:
                video_embeddings = torch.cat((video_embeddings, video_embedding), dim=0)
                seeg_embeddings = torch.cat((seeg_embeddings, seeg_embedding), dim=0)

        # Compute similarity
        sim = (video_embedding @ seeg_embedding.transpose(1, 0)) * math.e
        labels = torch.arange(video_embeddings.shape[0]).to(device)

        # Compute accuracy
        acc1, acc2 = compute_top_k_acc(sim, labels, top_k=[1, 2])

        if split == 'val':
            writer.add_scalar(f'Val/Acc@1 for Each Epoch', acc1, epoch + 1)
            writer.add_scalar(f'Val/Acc@2 for Each Epoch', acc2, epoch + 1)

        print(f'{split} Acc@1 {acc1:.3f}')
        print(f'{split} Acc@2 {acc2:.3f}')
        return acc1, acc2


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
        pred (`torch.Tensor`): The predicted labels.
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
