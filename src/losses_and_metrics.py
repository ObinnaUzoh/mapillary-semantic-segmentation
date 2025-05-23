import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def cross_entropy_plus_dice_loss(input:Tensor, target: Tensor, weights=(1, 1)):
    ce_loss = nn.BCEWithLogitsLoss()(input, target)
    di_loss = dice_loss(
        F.softmax(input, dim=1).float(),
        target,
        multiclass=True
    )
    return weights[0]*ce_loss + weights[1]*di_loss


CRITERIA = dict(cross_entropy_plus_dice_loss=cross_entropy_plus_dice_loss,
    cross_entropy_loss=cross_entropy_plus_dice_loss)