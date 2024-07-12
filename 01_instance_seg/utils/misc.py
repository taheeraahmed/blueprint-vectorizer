import numpy as np
import os
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def save_checkpoint(state, is_best, checkpoint='checkpoints/', filename='checkpoint.pth.tar', snapshot=None):
    """Saves checkpoint to disk"""
    # todo: also save the actual preds
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transfer_optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        if tensor.requires_grad:
            return tensor.detach().numpy()
        else:
            return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def dice_score(preds, labels, smooth=1e-6):
    """
    Calculate the Dice score.
    :param preds: predicted labels (binary or probabilities)
    :param labels: ground truth labels (binary)
    :param smooth: smoothing factor to avoid division by zero
    :return: Dice score
    """
    print(preds.shape, labels.shape)
    assert preds.shape == labels.shape, "Predictions and labels must have the same shape"
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection = (preds * labels).sum()
    return (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)
