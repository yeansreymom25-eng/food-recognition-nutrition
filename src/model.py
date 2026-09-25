import torch.nn as nn

from torchvision import models
from torchvision.models import MobileNet_V2_Weights


# --------------------------------------------------
# Create MobileNetV2
# --------------------------------------------------

def create_mobilenetv2(
    num_classes=20,
    pretrained=True,
    freeze_backbone=True
):

    # Load pretrained MobileNetV2
    if pretrained:

        weights = MobileNet_V2_Weights.DEFAULT

        model = models.mobilenet_v2(
            weights=weights
        )

    else:

        model = models.mobilenet_v2(
            weights=None
        )


    # --------------------------------------------------
    # Freeze pretrained feature layers
    # --------------------------------------------------

    if freeze_backbone:

        for param in model.features.parameters():
            param.requires_grad = False


    # --------------------------------------------------
    # Change classifier for food classes
    # --------------------------------------------------

    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model
