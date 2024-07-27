import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gin

# Datasets
@gin.configurable
def get_dataset(name='CIFAR100',
                train=True,
                transform_type='standard',
                size=224,
                normalize=True,
                flatten=False):
    dataset_classes = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }

    # Create transformations
    transformations = []
    if transform_type == 'augmented':
        transformations.extend([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
        ])
    else:
        transformations.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])

    transformations.append(torchvision.transforms.ToTensor())

    if normalize:
        transformations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    if flatten:
        transformations.append(transforms.Lambda(lambda x: torch.flatten(x)))

    transform = transforms.Compose(transformations)
                    
    root_dir = f'data/{name.lower()}'
    dataset = dataset_classes[name](root=root_dir, train=train, download=True, transform=transform)
    return dataset

# DataLoaders
@gin.configurable
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
