import albumentations
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json

def get_normalization_stats():
    output_dir = Path("./stats")
    output_file = output_dir / "synapse_formation_channel_stats.json"
    if not output_file.exists():
        raise FileNotFoundError(f"Normalization stats file not found at {output_file}")
    
    with open(output_file, "r") as f:
        channel_stats = json.load(f)

    means = [channel_stats[str(ch)]["mean"] for ch in range(len(channel_stats))]
    stds = [channel_stats[str(ch)]["std"] for ch in range(len(channel_stats))]

    return albumentations.Normalize(mean=means, std=stds)


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