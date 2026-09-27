"""Dataset loading and preprocessing utilities."""

import torch
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def get_data_loaders(dataset_name, batch_size, data_dir="./data"):
    """Create train, validation, and test loaders for a supported dataset."""
    if dataset_name in {"mnist", "fashion_mnist"}:
        return _get_image_loaders(dataset_name, batch_size, data_dir)
    if dataset_name in {"breast_cancer", "iris"}:
        return _get_tabular_loaders(dataset_name, batch_size)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _get_image_loaders(dataset_name, batch_size, data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST

    train_dataset = dataset_cls(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_cls(
        root=data_dir, train=False, download=True, transform=transform
    )

    # The 60k training set is split into 50k training and 10k validation samples.
    # A fixed generator makes the cleaned repository deterministic across reruns.
    split_generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        train_dataset, [50000, 10000], generator=split_generator
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


def _get_tabular_loaders(dataset_name, batch_size):
    if dataset_name == "iris":
        data = load_iris()
    else:
        data = load_breast_cancer()

    # Min-max scaling follows the experiment notebook.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data.data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(data.target, dtype=torch.long)
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Notebook split: 75% train, 5% validation, remaining 20% test.
    total_size = len(full_dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.05 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(101),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader
