import argparse
import logging
import os
import uuid
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from .evaluate import evaluate
from .models import UNet
from .losses_and_metrics import CRITERIA

from .dataset.mapillary_dataset import Mapillary_SemSeg_Dataset, N_LABELS

dir_checkpoint = Path(f'./checkpoints/{uuid.uuid4()}')

def train_and_val_one_epoch(model, 
    train_dataset, 
    train_loader, 
    val_loader,
    optimizer, 
    scheduler, 
    grad_scaler, 
    gradient_clipping,
    criterion, 
    experiment, 
    amp, 
    save_checkpoint,
    epoch):
    model.train()
    epoch_loss = 0
    epoch_total_samples = 0
    with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}', unit='img') as pbar:
        for images, true_masks in train_loader:
            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)

                loss = criterion(masks_pred, true_masks.float())

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            epoch_loss += loss.item()
            epoch_total_samples+= images.shape[0]
            pbar.set_postfix(**{'loss (batch)': loss.item()})

    experiment.log({
        'train loss': epoch_loss/epoch_total_samples,
        'epoch': epoch
    })
    
    # Evaluation round
    val_score = evaluate(model, val_loader, device, amp)
    scheduler.step(val_score)

    logging.info('Validation Dice score: {}'.format(val_score))
    experiment.log({
        'learning rate': optimizer.param_groups[0]['lr'],
        'validation Dice': val_score,
        'images': wandb.Image(images[0].cpu()),
        'masks': {
            'true': wandb.Image(true_masks.argmax(dim=1)[0].float().cpu()),
            'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
        },
        'epoch': epoch,
    })

    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f'Checkpoint {epoch} saved!')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        image_height: int = 512,
        image_width: int = 512,
        criterion: str = 'cross_entropy_plus_dice_loss',
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_dataset = Mapillary_SemSeg_Dataset('training', './data', (image_height, image_width), augment=True, verbose=False)
    val_dataset = Mapillary_SemSeg_Dataset('validation', './data', (image_height, image_width), augment=False, verbose=False)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count()//2, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    wandb.login()
    experiment = wandb.init(project='mapillary')
    experiment.config.update(
        dict(model='UNET', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, image_height=image_height, image_width=image_width, amp=amp, 
             dir_checkpoint=dir_checkpoint,
             criterion=criterion)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Image height:  {image_height}
        Image width:  {image_width}
        Mixed Precision: {amp}
        Checkpoint dir: {dir_checkpoint}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    try:
        criterion_obj = CRITERIA[criterion]
    except keyError:
        logger.error(f'Unimplemented criterion {criterion}')
        raise keyError

    train_val_args = dict(model=model, 
        train_dataset=train_dataset, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        grad_scaler=grad_scaler,  
        gradient_clipping=gradient_clipping, 
        criterion=criterion_obj, 
        experiment=experiment, 
        amp=amp, 
        save_checkpoint=save_checkpoint)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        train_and_val_one_epoch(**train_val_args, epoch=epoch)
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--image-height', '-he', type=int, default=512, help='Image height')
    parser.add_argument('--image-width', '-wi', type=int, default=512, help='Image width')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--criterion', '-c', type=str, default='cross_entropy_plus_dice_loss', help='loss function for training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=N_LABELS, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            image_height=args.image_height,
            image_width=args.image_width,
            criterion=args.criterion,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        raise RuntimeError('CUDA OOM')
        