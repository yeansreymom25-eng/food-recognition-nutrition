import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def get_transforms():
    """
    Create image transformations for training,
    validation, and testing.
    """

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

    return train_transform, val_test_transform


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

def load_datasets(data_dir):
    """
    Load train, validation, and test datasets.

    Expected structure:

    data_dir/
        train/
        validation/
        test/
    """

    train_transform, val_test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "validation"),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
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
    data_dir,
    batch_size=32,
    num_workers=2
):
    """
    Create PyTorch DataLoaders.
    """

    train_dataset, validation_dataset, test_dataset = (
        load_datasets(data_dir)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes
    )


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

def print_dataset_info(data_dir):
    """
    Display dataset sizes and class names.
    """

    train_dataset, validation_dataset, test_dataset = (
        load_datasets(data_dir)
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))
    print("Number of classes:", len(train_dataset.classes))

    print("\nClasses:")
    for class_name in train_dataset.classes:
        print("-", class_name)

# get_transforms() handles resize, augmentation and normalization.
# load_datasets() loads train, validation and test folders using ImageFolder.
# create_dataloaders() creates batches for PyTorch.
