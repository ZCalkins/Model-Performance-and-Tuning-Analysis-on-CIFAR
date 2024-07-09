import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gin

# Data Transformation
@gin.configurable
def create_transform(transform_type='standard', size=224, normalize=True, flatten=False):
    transformations = []
    
    if transform_type == 'augmented':
        transformations.extend([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
        ])
    else:
        transformations.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])
    
    transformations.append(transforms.ToTensor())
    
    if normalize:
        transformations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    if flatten:
        transformations.append(transforms.Lambda(lambda x: torch.flatten(x)))
    
    return transforms.Compose(transformations)

# Datasets
@gin.configurable
def get_dataset(name='CIFAR100', train=True, transform=None, transform_config=None):
    dataset_classes = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }

    if transform_config is not None and isinstance(transform_config, dict):
        transform = create_transform(**transform_config)
    else transform is None:
        transform = create_transform()
    
    root_dir = f'data/{name.lower()}'
    dataset = dataset_classes[name](root=root_dir, train=train, download=True, transform=transform)
    return dataset

# DataLoaders
@gin.configurable
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
