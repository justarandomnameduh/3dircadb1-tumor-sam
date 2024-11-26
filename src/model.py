# src/model.py

import segmentation_models_pytorch as smp
import torch.nn as nn

def get_unet_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1):
    """
    Initializes and returns a U-Net model.
    """
    model = smp.Unet(
        encoder_name=encoder_name,        # Encoder: ResNet-34
        encoder_weights=encoder_weights,  # Use ImageNet pre-trained weights
        in_channels=in_channels,          # Input channels (1 for grayscale)
        classes=classes                   # Output channels (1 for binary segmentation)
    )
    return model
