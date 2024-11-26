# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bars

def dice_coefficient(preds, targets, smooth=1e-6):
    """
    Calculates the Dice Coefficient.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + smooth)
    return dice.mean().item()

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    Reference:
        https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    """
    Trains the U-Net model.
    Returns training loss and Dice coefficient history.
    """
    model.train()
    train_loss_history = []
    train_dice_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_dice = 0.0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch in loop:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            dice = dice_coefficient(outputs, masks)
            running_dice += dice * images.size(0)

            loop.set_postfix(loss=loss.item(), dice=dice)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_dice_history.append(epoch_dice)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice Coef: {epoch_dice:.4f}")

    return train_loss_history, train_dice_history
