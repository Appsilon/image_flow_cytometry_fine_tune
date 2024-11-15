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
sys.path.append('..')
from utils import data_module, resnet
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping

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
reshape_size = 256
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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=6)


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
model = resnet.ResnetModel(len(set_of_interesting_classes), len(selected_channels))

# %%
lr=0.01
batch_size=128
max_epochs=1000

# %%
module = data_module.SynapseFormationDataModule(metadata, train_index, validation_index, test_index, label_map, selected_channels, statistics, train_transform,
                                                validation_transform, test_transform, batch_size, reshape_size)

# %%
run = neptune.init_run(
    project="appsilon/image-flow-cytometry-finetune",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OTA1ZjQwZS03MDczLTRiMzgtYmRhOS1iYjM2Y2EyMjcwMDMifQ==",
)

# %%
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=pl.loggers.NeptuneLogger(run=run, log_model_checkpoints=False),
    callbacks=[lr_monitor])

trainer.fit(model, datamodule=module)

trainer.test(model, datamodule=module)

run.stop()

# %%


# #resnet18_modified.load_state_dict(torch.load('supervised_learning_synapse_model.pth')) 

# #lr_scheduler = LRScheduler(policy='StepLR', step_size=5, gamma=0.6)
# lr_scheduler = LRScheduler(policy='ReduceLROnPlateau', factor=0.1, patience=10)
# #checkpoint = Checkpoint(f_params='resnet_18_imagenet_pretraiend_supervised_learning.pth', monitor='valid_acc_best')


# epoch_scoring = EpochScoring("f1_macro", 
#                              name =  "valid_f1_macro", 
#                              on_train = False,
#                              lower_is_better = False)

# early_stopping = EarlyStopping(monitor='valid_f1_macro', 
#                                patience=100, 
#                                threshold=0.0001, 
#                                threshold_mode='rel', 
#                                lower_is_better=False)

# model = NeuralNetClassifier(    
#     swin, 
#     criterion=nn.CrossEntropyLoss,
#     lr=0.01,
#     batch_size=128,
#     max_epochs=1000,
#     optimizer=optim.Adam,
#     iterator_train__shuffle=True,
#     iterator_train__num_workers=4,
#     iterator_valid__shuffle=False,
#     iterator_valid__num_workers=2,
#     callbacks=[lr_scheduler,epoch_scoring, early_stopping],
#     train_split=predefined_split(validation_dataset_resnet_18),
#     device="cuda",
#     warm_start=True)


