# src/evaluate.py

import torch

def dice_coefficient(preds, targets, smooth=1e-6):
    """
    Calculates the Dice Coefficient.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + smooth)
    return dice.mean().item()

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the U-Net model on the test set.
    Returns average loss and Dice coefficient.
    """
    model.eval()
    test_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            test_loss += loss.item() * images.size(0)
            dice = dice_coefficient(outputs, masks)
            dice_scores.append(dice * images.size(0))

    average_loss = test_loss / len(test_loader.dataset)
    average_dice = sum(dice_scores) / len(test_loader.dataset)
    print(f"Test Loss: {average_loss:.4f}, Test Dice Coef: {average_dice:.4f}")
    return average_loss, average_dice
