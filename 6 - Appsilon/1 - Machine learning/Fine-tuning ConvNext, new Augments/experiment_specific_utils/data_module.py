from torch.utils.data import DataLoader
from lightning import LightningDataModule
from experiment_specific_utils.dataset import DatasetGenerator

class SynapseFormationDataModule(LightningDataModule):
    def __init__(self, metadata, train_index, val_index, test_index, label_map, selected_channels,
                 train_transform, val_transform, test_transform, batch_size=128, reshape_size=160, skip_crop_pad=False):
        super().__init__()
        self.metadata = metadata
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index
        self.label_map = label_map
        self.selected_channels = selected_channels
        self.batch_size = batch_size
        self.reshape_size = reshape_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.skip_crop_pad = skip_crop_pad

    def setup(self, stage=None):
        print("Initializing datasets...")
        self.train_dataset = DatasetGenerator(self.metadata.loc[self.train_index, :], label_map=self.label_map,
                                              selected_channels=self.selected_channels, reshape_size=self.reshape_size, 
                                              transform=self.train_transform, skip_crop_pad=self.skip_crop_pad)

        self.val_dataset = DatasetGenerator(self.metadata.loc[self.val_index, :], label_map=self.label_map,
                                            selected_channels=self.selected_channels, reshape_size=self.reshape_size, 
                                            transform=self.val_transform, skip_crop_pad=self.skip_crop_pad)

        self.test_dataset = DatasetGenerator(self.metadata.loc[self.test_index, :], label_map=self.label_map,
                                             selected_channels=self.selected_channels, reshape_size=self.reshape_size, 
                                             transform=self.test_transform, skip_crop_pad=self.skip_crop_pad)
        print("Datasets initialized successfully!")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)