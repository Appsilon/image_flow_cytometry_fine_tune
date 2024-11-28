# %%
import pandas as pd
import numpy as np
import torch
import neptune
import os
import random
import lightning.pytorch as pl
import sys
from pathlib import Path
sys.path.append(str(Path('../../').resolve()))
from utils import convnext, tools
from lightning.pytorch.callbacks import LearningRateMonitor
from fastai.vision.all import *
import albumentations
from albumentations.pytorch import ToTensorV2
from experiment_specific_utils import data_module, transforms

# %%

seed_value = 42

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)

np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

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
reshape_size = 256
train_transform = transforms.train_transform_fit_image(reshape_size, include_normalization = True)
test_val_transform = transforms.test_val_transform_fit_image(reshape_size)

# %%
lr=0.0004
batch_size=32
max_epochs=50

# %%
print("Available cuda memory before model initialization: ")
tools.print_cuda_memory()

synapse_formation_module = data_module.SynapseFormationDataModule(metadata, train_index, validation_index, test_index, label_map, selected_channels, train_transform,
                                                test_val_transform, test_val_transform, batch_size, reshape_size, skip_crop_pad=True)

synapse_formation_module.setup(stage='fit')
train_loader = synapse_formation_module.train_dataloader()
val_loader = synapse_formation_module.val_dataloader()
model = convnext.ConvnextModel(num_classes=len(set_of_interesting_classes), in_chans=len(selected_channels), steps_per_epoch=len(train_loader), learning_rate=lr, max_epochs=max_epochs)

tools.save_multichannel_preview(train_loader, n_samples=10, save_path="train_multichannel_convnext_fit_image_no_pad_preview.png")
tools.save_multichannel_preview(val_loader, n_samples=10, save_path="valid_multichannel_convnext_fit_image_no_pad_preview.png")
print("Preview saved!\n\n")
# %%
run = neptune.init_run(
    project="appsilon/image-flow-cytometry-finetune",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OTA1ZjQwZS03MDczLTRiMzgtYmRhOS1iYjM2Y2EyMjcwMDMifQ==",
)
run["sys/notes"] = "ConvNext without padding, instead resizing and center cropping"
# %%
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=pl.loggers.NeptuneLogger(run=run, log_model_checkpoints=False),
    callbacks=[lr_monitor])

print("\nAvailable cuda memory before Neptune run: ", tools.print_cuda_memory())

trainer.fit(model, datamodule=synapse_formation_module)

trainer.test(model, datamodule=synapse_formation_module)

run.stop()


