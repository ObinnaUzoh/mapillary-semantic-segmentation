import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for image, mask_true in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            mask_true = mask_true.float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring unlabeled 
            dice_score += multiclass_dice_coeff(mask_pred[:, :-1], mask_true[:, :-1], reduce_batch_first=False)
         
    net.train()
    return dice_score / max(num_val_batches, 1)