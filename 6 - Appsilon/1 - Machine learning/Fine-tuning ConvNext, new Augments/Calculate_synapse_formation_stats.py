import pandas as pd
import numpy as np
import torch
import os
import random
import sys
from pathlib import Path
sys.path.append(str(Path('../../').resolve()))
from fastai.vision.all import *
from experiment_specific_utils import data_module, transforms
import json

seed_value = 42

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)

np.random.seed(seed_value)
torch.manual_seed(seed_value)

metadata = pd.read_csv("/home/jedrzej/projects/image_flow_cytometry_fine_tune/data/jedrzej/metadata_subset.csv.gz")
metadata

indx = metadata.condition.isin(["-SEA","+SEA"])
metadata = metadata.loc[indx, :].reset_index(drop = True )

set_of_interesting_classes = ['B_cell',  'T_cell', 
                        'T_cell_with_signaling',
                        'T_cell_with_B_cell_fragments',
                        'B_T_cell_in_one_layer',
                        'Synapses_without_signaling', 
                        'Synapses_with_signaling',
                        'No_cell_cell_interaction', 
                        'Multiplets'] 

indx = metadata.set.isin([ "train", "validation","test" ])
indx = indx & metadata.label.isin(set_of_interesting_classes)

train_index = metadata["set"] == "train"
train_index = train_index & metadata.label.isin(set_of_interesting_classes)
train_index = train_index[train_index].index

validation_index = metadata["set"] == "validation"
validation_index = validation_index & metadata.label.isin(set_of_interesting_classes)
validation_index = validation_index[validation_index].index

test_index = metadata["set"] == "test"
test_index = test_index & metadata.label.isin(set_of_interesting_classes)
test_index = test_index[test_index].index

label_map = dict()
for i, cl in enumerate(set_of_interesting_classes):
    label_map[cl] = i

label_map['-1'] = -1
label_map[-1] = -1

selected_channels = [0,3,4,5,6]
model_dir = "models"
log_dir = "logs"
scaling_factor = 4095.
reshape_size = 256
lr=0.0004
batch_size=32
max_epochs=10
train_transform = transforms.train_transform(reshape_size, include_normalization=False)
test_val_transform = None # Doesn't matter here, we only need train preprocessing to calculate stats

synapse_formation_module = data_module.SynapseFormationDataModule(metadata, train_index, validation_index, test_index, label_map, selected_channels, train_transform,
                                                test_val_transform, test_val_transform, batch_size, reshape_size)

synapse_formation_module.setup(stage='fit')
train_loader = synapse_formation_module.train_dataloader()

all_images = []
for images, labels in train_loader:
    all_images.append(images)

all_images = torch.cat(all_images, dim=0)  # Shape: [num_samples, num_channels, height, width]

print(f"All images shape: {all_images.shape}")

channel_means = all_images.mean(dim=[0, 2, 3])
channel_stds = all_images.std(dim=[0, 2, 3])

print("Channel-wise Mean and Std:")
for ch, (mean, std) in enumerate(zip(channel_means, channel_stds)):
    print(f"Channel {ch} - Mean: {mean:.4f}, Std: {std:.4f}")

channel_stats = {ch: {"mean": mean.item(), "std": std.item()} for ch, (mean, std) in enumerate(zip(channel_means, channel_stds))}
output_file = "synapse_formation_channel_stats.json"
with open(output_file, "w") as f:
    json.dump(channel_stats, f, indent=4)

print(f"Channel statistics saved to {output_file}")