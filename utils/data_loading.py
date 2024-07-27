import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2
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
            v2.RandomResizedCrop(size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.AutoAugment(),
        ])
    else:
        transformations.extend([
            v2.Resize(size),
            v2.CenterCrop(size),
        ])

    transformations.append(v2.compose([v2.ToImageTensor(), v2.ConvertImageDType()]))

    if normalize:
        transformations.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    if flatten:
        transformations.append(v2.Lambda(lambda x: torch.flatten(x)))

    transform = v2.Compose(transformations)
                    
    root_dir = f'data/{name.lower()}'
    dataset = dataset_classes[name](root=root_dir, train=train, download=True, transform=transform)
    return dataset

# DataLoaders
@gin.configurable
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
