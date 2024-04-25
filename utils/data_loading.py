# data transformations

flat_transform_augment = transforms.Compose([
  v2.RandomResizedCrop(224),
  v2.RandomHorizontalFlip(),
  v2.AutoAugment(),
  v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  transforms.Lambda(lambda x: torch.flatten(x))
])

std_img_transform = transforms.Compose([
  v2.RandomResizedCrop(224),
  v2.RandomHorizontalFlip(),
  v2.AutoAugment(),
  v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

flat_transform_basic = transforms.Compose([
  v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
  transforms.Lambda(lambda x: torch.flatten())
])

# DataLoaders

train_dataloader_flat = DataLoader(dataset=train_data_flat,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

train_dataloader_flat_augment = DataLoader(dataset=train_data_flat_augment,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

train_dataloader_std_img = DataLoader(dataset=train_data_std_img,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)

train_dataloader_std_img_augment = DataLoader(train_data_std_img_augment,
                                              batch_size_BATCH_SIZE,
                                              shuffle=True)

test_dataloader_flat = DataLoader(dataset=test_data_flat,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

test_dataloader_std_img = DataLoader(dataset=test_data_std_img,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
