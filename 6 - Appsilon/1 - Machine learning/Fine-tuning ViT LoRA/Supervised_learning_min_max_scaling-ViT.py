# %%
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from scifAI.dl.dataset import DatasetGenerator
from scifAI.dl.utils import get_statistics
from torch.utils.data import DataLoader
import neptune
import os
import random
import lightning.pytorch as pl
import sys
from pathlib import Path
sys.path.append(str(Path('../../').resolve()))
from utils import data_module, vit_lora, tools
from lightning.pytorch.callbacks import LearningRateMonitor
from fastai.vision.all import *

# %%

seed_value = 42

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)

np.random.seed(seed_value)
torch.manual_seed(seed_value)

# %%
metadata = pd.read_csv("/home/jedrzej/projects/image_flow_cytometry_fine_tune/data/jedrzej/metadata_subset.csv.gz")
metadata

# %%
metadata.set.unique()

# %%
indx = metadata.condition.isin(["-SEA","+SEA"])
metadata = metadata.loc[indx, :].reset_index(drop = True )

# %%
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

# %%
metadata["set"].unique()

# %%
label_map = dict()
for i, cl in enumerate(set_of_interesting_classes):
    label_map[cl] = i

label_map['-1'] = -1
label_map[-1] = -1


# %%
label_map

# %%
channels = {
     "Ch1": ("Greys", "BF"),  
     "Ch2": ("Greens", "Antibody"),
     "Ch3": ("Reds", "CD18"),
     "Ch4": ("Oranges", "F-Actin"),
     "Ch6": ("RdPu", "MHCII"),
     "Ch7": ("Purples", "CD3/CD4"),
     "Ch11": ("Blues", "P-CD3zeta"),
     "Ch12": ("Greens", "Live-Dead")
 }

# %%
selected_channels = [0,3,4,5,6]
model_dir = "models"
log_dir = "logs"
scaling_factor = 4095.
reshape_size = 224
train_transform = [
         transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45)
        ]
test_transform = [ ]

# %%
train_dataset = DatasetGenerator(metadata=metadata.loc[train_index,:],
                                 label_map=label_map,
                                 selected_channels=selected_channels,
                                 scaling_factor=scaling_factor,
                                 reshape_size=reshape_size,
                                 transform=transforms.Compose(train_transform))

# %%
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers = 4)


# %%
statistics = get_statistics(train_loader, selected_channels=selected_channels)


# %%
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# %%
class MinMaxScaler(object):
    def __init__(self, min_in , max_in, min_out, max_out):
        self.min_in = min_in.reshape(-1,1,1)
        self.max_in = max_in.reshape(-1,1,1)
        self.min_out = min_out
        self.max_out = max_out
        
    def __call__(self, tensor):
        
        tensor_ = (tensor - self.min_in)/(self.max_in - self.min_in)
        tensor_ = tensor_*(self.max_out - self.min_out) + self.min_out
        tensor_[tensor_<self.min_out]= self.min_out
        tensor_[tensor_>self.max_out]= self.max_out
        return tensor_
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_out={0}, max_out={1})'.format(self.min_out, self.max_out)

# %%
train_transform = transforms.Compose([ 
        MinMaxScaler(           min_in =  statistics["p05"] , 
                                max_in =  statistics["p95"] , 
                                min_out =  0. , 
                                max_out =  1.),
        transforms.RandomResizedCrop(reshape_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AddGaussianNoise(mean=0., std=0.01),
])

validation_transform =  transforms.Compose([ 
        MinMaxScaler(           min_in =  statistics["p05"] , 
                                max_in =  statistics["p95"] , 
                                min_out =  0. , 
                                max_out =  1.),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AddGaussianNoise(mean=0., std=0.01),
])

test_transform =  transforms.Compose([ 
        MinMaxScaler(           min_in =  statistics["p05"] , 
                                max_in =  statistics["p95"] , 
                                min_out =  0. , 
                                max_out =  1.),
])


# %%
lr=0.0004
batch_size=32
max_epochs=10
lora_r_alpha = 64
lora_target_modules = target_modules = [
    "attention.query",  # Query layer in attention
    "attention.key",    # Key layer in attention
    "attention.value",  # Value layer in attention
    "attention.output.dense",  # Dense layer in attention output
    "intermediate.dense",  # Intermediate MLP layer
    "output.dense"  # Output MLP layer
]
lora_bias = "none"
lora_dropout = 0.1

# %%
from fastai.vision.all import *
from torch.utils.data import DataLoader

train_dataset = DatasetGenerator(metadata=metadata.loc[train_index, :],
                                 label_map=label_map,
                                 selected_channels=selected_channels,
                                 scaling_factor=scaling_factor,
                                 reshape_size=reshape_size,
                                 transform=train_transform)

valid_dataset = DatasetGenerator(metadata=metadata.loc[validation_index, :],
                                 label_map=label_map,
                                 selected_channels=selected_channels,
                                 scaling_factor=scaling_factor,
                                 reshape_size=reshape_size,
                                 transform=validation_transform)

# Convert to FastAI DataLoaders
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
dls = DataLoaders(train_dl, valid_dl)
print(tools.print_cuda_memory())
# %%
model = vit_lora.VitModel(num_classes=len(set_of_interesting_classes), in_chans=len(selected_channels), steps_per_epoch=len(train_loader), learning_rate=lr, max_epochs=max_epochs,
                          lora_r_alpha=lora_r_alpha, lora_target_modules=lora_target_modules, lora_dropout=lora_dropout, lora_bias=lora_bias)
# %%
module = data_module.SynapseFormationDataModule(metadata, train_index, validation_index, test_index, label_map, selected_channels, statistics, train_transform,
                                                validation_transform, test_transform, batch_size, reshape_size)

# %%
print(tools.print_cuda_memory())

run = neptune.init_run(
    project="appsilon/image-flow-cytometry-finetune",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OTA1ZjQwZS03MDczLTRiMzgtYmRhOS1iYjM2Y2EyMjcwMDMifQ==",
)

# %%
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=pl.loggers.NeptuneLogger(run=run, log_model_checkpoints=False),
    callbacks=[lr_monitor])
print(tools.print_cuda_memory())

trainer.fit(model, datamodule=module)

trainer.test(model, datamodule=module)

run.stop()


