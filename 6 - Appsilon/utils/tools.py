import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random

def print_cuda_memory():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # Convert to MB
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # Convert to MB
        free_memory = total_memory - reserved

        print(f"Device {i}:")
        print(f"  Allocated Memory: {allocated:.2f} MB")
        print(f"  Reserved Memory: {reserved:.2f} MB")
        print(f"  Free Memory: {free_memory:.2f} MB")
        print(f"  Total Memory: {total_memory:.2f} MB")

def save_multichannel_preview(data_loader, n_samples=5, save_path="multichannel_preview.png"):
    """
    Save a preview of the dataset where each column represents a sample 
    with its corresponding channels displayed vertically.
    
    Parameters:
    - data_loader: DataLoader to fetch the batch from.
    - n_samples: Number of random samples to visualize.
    - save_path: Path to save the output image.
    """
    batch = next(iter(data_loader))
    images, labels = batch
    
    n_samples = min(n_samples, images.size(0))
    selected_indices = random.sample(range(images.size(0)), n_samples)
    
    fig, axes = plt.subplots(nrows=images.size(1), ncols=n_samples, figsize=(n_samples * 3, images.size(1) * 3))
    fig.subplots_adjust(hspace=0.3)

    for col, idx in enumerate(selected_indices):
        for row in range(images.size(1)):
            ax = axes[row, col] if images.size(1) > 1 else axes[col]
            img = images[idx, row].cpu().numpy()
            
            ax.imshow(img, cmap="gray")
            ax.axis('off')
            
            if row == 0:
                ax.set_title(f"Index {idx}", fontsize=10, pad=15)

    plt.savefig(save_path)
    plt.close()