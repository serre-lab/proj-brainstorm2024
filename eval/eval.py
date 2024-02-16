import torch
from tqdm import tqdm
from util.loss import recon_loss, general_contrast_loss, agg_loss, contrastive_loss
import matplotlib.pyplot as plt
import wandb


def val_autoencoder(autoencoder, eval_loader, device, alpha_value):
    autoencoder.eval()

    embeds = None
    labels = None

    recon_loss_meter = AverageMeter()

    with torch.no_grad():
        for seeg, video_idx, phase in tqdm(eval_loader):
            batch_size = seeg.size(0)

            seeg = seeg.to(device)
            video_idx = video_idx.to(device)

            # Forward
            seeg_recon, embed = autoencoder(seeg)
            embed = embed.flatten(start_dim=1)

            if embeds is None:
                embeds = embed
                labels = video_idx
            else:
                embeds = torch.cat((embeds, embed), dim=0)
                labels = torch.cat((labels, video_idx), dim=0)

            r_loss = recon_loss(seeg, seeg_recon)
            recon_loss_meter.update(r_loss.item(), batch_size)

        # Compute similarity
        c_loss = general_contrast_loss(embeds, labels)
        total_loss = agg_loss(recon_loss_meter.avg, c_loss, alpha_value)

        wandb.log({"val_total_loss": total_loss,
                   "val_recon_loss": recon_loss_meter.avg,
                   "val_contra_loss": c_loss,
                   "val_scaled_contra_loss": c_loss * alpha_value})

        print(f'Reconstruction Loss: {recon_loss_meter.avg:.4f}')
        print(f'Contrastive Loss: {c_loss:.4f}')
        print(f'Scaled Contrastive Loss: {c_loss*alpha_value:.4f}')
        print(f'Total Loss: {total_loss:.4f}')
        return recon_loss_meter.avg, c_loss, total_loss


def val_classifier(autoencoder, classifier, eval_loader, device):
    with torch.no_grad():
        autoencoder.eval()
        classifier.eval()

        preds = None
        labels = None

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

        loss = torch.nn.CrossEntropyLoss()(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean().item()
        wandb.log({"clf_val_loss": loss})
        wandb.log({"clf_val_acc": acc})
        print(f'Classification Loss: {loss:.4f}')
        print(f'Classification Acc: {acc:.4f}')
        return acc


def val_e2e_classifier(e2e_classifier, eval_loaders, device, alpha_value):
    e2e_classifier.eval()

    recon_loss_meter = AverageMeter()
    cls_acc_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for id, eval_loader in enumerate(eval_loaders):
        with torch.no_grad():
            for seeg, video_idx, phase in tqdm(eval_loader):
                batch_size = seeg.size(0)

                seeg = seeg.to(device)
                video_idx = video_idx.to(device)

                # Forward
                logits, seeg_recon = e2e_classifier(seeg, id)

                # Compute reconstruction loss
                r_loss = recon_loss(seeg, seeg_recon)
                recon_loss_meter.update(r_loss.item(), batch_size)

                # Compute classification accuracy
                cls_acc = (logits.argmax(dim=1) == video_idx).float().mean().item()
                cls_acc_meter.update(cls_acc, batch_size)

                # Compute classification loss
                cls_loss = torch.nn.CrossEntropyLoss()(logits, video_idx)
                cls_loss_meter.update(cls_loss.item(), batch_size)

                # Compute total loss
                total_loss = agg_loss(r_loss, cls_loss, alpha_value)
                total_loss_meter.update(total_loss, batch_size)

            wandb.log({"val_recon_loss": recon_loss_meter.avg,
                       "val_cls_acc": cls_acc_meter.avg,
                       "val_cls_loss": cls_loss_meter.avg,
                       "val_scaled_cls_loss": cls_loss_meter.avg * alpha_value,
                       "val_total_loss": total_loss_meter.avg})

        print(f'Reconstruction Loss: {recon_loss_meter.avg:.4f}')
        print(f'Classification Acc: {cls_acc_meter.avg:.4f}')
        return cls_acc_meter.avg


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
