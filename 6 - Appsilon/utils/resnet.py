import torch.nn as nn
from torchvision.models import resnet152, resnet18, resnet34, resnet50
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from timm import create_model
from torchmetrics import Accuracy, F1Score

class ResnetModel(LightningModule):
    def __init__(self, num_classes, in_chans, learning_rate=0.01):
        super().__init__()
        model = resnet18(pretrained=True)
        self.save_hyperparameters()
        if in_chans != 3:
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = model

        self.learning_rate = learning_rate
        self.train_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1_score = F1Score(num_classes=num_classes, task="multiclass", average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc(outputs, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(outputs, labels), prog_bar=True)
        self.log("val_f1", self.f1_score(outputs, labels), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("test_loss", loss)
        self.log("test_acc", self.val_acc(outputs, labels))
        self.log("test_f1", self.f1_score(outputs, labels))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_f1"}
