import torch
from tqdm import tqdm
from util.loss import recon_loss, general_contrast_loss, agg_loss
import wandb


def val_autoencoder(model, eval_loader, device, alpha_value):
    model.eval()

    embeds = None
    labels = None

    recon_loss_meter = AverageMeter()

    with torch.no_grad():
        for seeg, video_idx, phase in tqdm(eval_loader):
            batch_size = seeg.size(0)

            seeg = seeg.to(device)
            video_idx = video_idx.to(device)

            # Forward
            seeg_recon, embed_before = model(seeg)
            embed = embed_before.flatten(start_dim=1)

            if embeds is None:
                embeds = embed
                labels = video_idx
            else:
                embeds = torch.cat((embeds, embed), dim=0)
                labels = torch.cat((labels, video_idx), dim=0)

            r_loss = recon_loss(seeg, seeg_recon)

            with torch.no_grad():
                recon_loss_meter.update(r_loss.item(), batch_size)

        # Compute similarity
        c_loss = general_contrast_loss(embeds, labels) / len(eval_loader)
        total_loss = agg_loss(recon_loss_meter.avg, c_loss, alpha_value)

        wandb.log({"classifier_val_loss": total_loss,
                   "classifier_val_recon_loss": recon_loss_meter.avg,
                   "classifier_val_contra_loss": c_loss.item(),
                   "classifier_val_scaled_contra_loss": c_loss.item() * alpha_value})

        print(f'Recontruction Loss: {recon_loss_meter.avg:.4f}')
        print(f'Scaled Contrastive Loss: {c_loss*alpha_value:.4f}')
        print(f'Total Loss: {total_loss:.4f}')
        return recon_loss_meter.avg, c_loss, total_loss


def val_classifier(autoencoder, classifier, eval_loader, device):
    autoencoder.eval()
    classifier.eval()

    preds = None
    labels = None

    with torch.no_grad():
        for seeg, video_idx, phase in tqdm(eval_loader):
            seeg = seeg.to(device)
            video_idx = video_idx.to(device)

            _, embed = autoencoder(seeg)
            output = classifier(embed)

            if preds is None:
                preds = output
                labels = video_idx
            else:
                preds = torch.cat((preds, output), dim=0)
                labels = torch.cat((labels, video_idx), dim=0)

        acc = (preds.argmax(dim=1) == labels).float().mean().item()
        wandb.log({"classifier_val_accuracy": acc})
        print(f'Classification Accuracy: {acc:.4f}')
        return acc



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
