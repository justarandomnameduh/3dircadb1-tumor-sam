# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_history(train_loss, train_dice, num_epochs):
    """
    Plots the training loss and Dice coefficient over epochs.
    """
    plt.figure(figsize=(12,5))

    # Plot Training Loss
    plt.subplot(1,2,1)
    plt.plot(range(1, num_epochs+1), train_loss, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot Training Dice Coefficient
    plt.subplot(1,2,2)
    plt.plot(range(1, num_epochs+1), train_dice, label='Training Dice Coef', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coef')
    plt.title('Training Dice Coefficient Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_segmentation(model, dataset, device, num_samples=5):
    """
    Visualizes segmentation results on random samples from the dataset.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)  # Shape: (1, 1, H, W)
        mask = sample['mask'].squeeze().cpu().numpy()    # Shape: (H, W)

        with torch.no_grad():
            output = model(image)
            preds = torch.sigmoid(output).squeeze().cpu().numpy()

        preds_binary = (preds > 0.5).astype(np.uint8)
        dice = (2 * (preds_binary * mask).sum()) / ((preds_binary + mask).sum() + 1e-6)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sample['image'].squeeze(), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # axes[2].imshow(sample['image'].squeeze(), cmap='gray')
        axes[2].imshow(preds_binary, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Predicted Mask\nDice Coef: {dice:.4f}')
        axes[2].axis('off')

        plt.show()
