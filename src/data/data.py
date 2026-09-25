import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Image preprocessing and augmentation
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

def load_datasets(data_path):

    train_dataset = datasets.ImageFolder(
        os.path.join(
            data_path,
            "train"
        ),
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        os.path.join(
            data_path,
            "validation"
        ),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(
            data_path,
            "test"
        ),
        transform=val_test_transform
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

def create_dataloaders(
    data_path,
    batch_size=32
):

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = load_datasets(
        data_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes
    )
