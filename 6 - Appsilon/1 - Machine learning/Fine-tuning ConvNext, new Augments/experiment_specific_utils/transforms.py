import albumentations
from albumentations.pytorch import ToTensorV2

def get_normalization_stats():
    pass

def train_transform(reshape_size, include_normalization=True):
    train_transforms = [
    # albumentations.SmallestMaxSize(max_size=reshape_size),
    # albumentations.CenterCrop(height=reshape_size, width=reshape_size),
    albumentations.VerticalFlip(p=0.5),         
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(limit=90, p=0.5),
    albumentations.RandomResizedCrop(height=reshape_size, width=reshape_size, scale=(0.8, 1), p=0.5),
    # albumentations.GaussianNoise(var_limit=(10.0, 50.0), p=1.0), not noise for now
    ]
    if include_normalization:
        norm_stats = get_normalization_stats()
        train_transforms.append(norm_stats)
    train_transforms.append(ToTensorV2())
    return albumentations.Compose(train_transforms)

def test_val_transform():
    norm_stats = get_normalization_stats()
    return albumentations.Compose([norm_stats, ToTensorV2()])