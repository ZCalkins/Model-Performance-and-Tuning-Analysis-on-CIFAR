# Data Transformation

import gin
from torchvision.transforms import v2

@gin.configurable
def create_transform(transform_type='standard', size=224, normalize=True, flatten=False):

    transformations = []
    if transform_type == 'augmented':
        transformations.extend([
            v2.RandomResizedCrop(size),
            v2.RandomHorizontalFlip(),
            v2.AutoAugment(),
        ])
    else:
        transformations.extend([
            v2.Resize(size),
            v2.CenterCrop(size),
        ])
    
    transformations.append(v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    
    if normalize:
        transformations.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    if flatten:
        transformations.append(transforms.Lambda(lambda x: torch.flatten(x)))
    
    return transforms.Compose(transformations)

# Datasets

@gin.configurable
def get_dataset(name='CIFAR100', train=True, transform_config=None):
    dataset_classes = {
        'CIFAR100': datasets.CIFAR100
    }
    transform = get_transform(**transform_config) if transform_config else None
    dataset = dataset_classes[name](root='./data', train=train, download=True, transform=transform)
    return dataset

# DataLoaders

@gin.configurable
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
