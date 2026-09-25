import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def create_mobilenetv2(
    num_classes=20,
    pretrained=True,
    freeze_backbone=True,
    unfreeze_last_blocks=0
):
    """
    Create MobileNetV2 for food classification.

    Args:
        num_classes: Number of food classes.
        pretrained: Use ImageNet pretrained weights.
        freeze_backbone: Freeze feature layers.
        unfreeze_last_blocks: Number of last feature blocks to unfreeze.

    Returns:
        PyTorch MobileNetV2 model.
    """

    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    model = models.mobilenet_v2(
        weights=weights
    )

    # Replace original classifier
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Fine-tuning option
    if unfreeze_last_blocks > 0:

        for layer in model.features[
            -unfreeze_last_blocks:
        ]:
            for param in layer.parameters():
                param.requires_grad = True

    return model
