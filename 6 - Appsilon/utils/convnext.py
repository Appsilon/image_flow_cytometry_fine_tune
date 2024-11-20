import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from timm import create_model
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import OneCycleLR

class ConvnextModel(LightningModule):
    def __init__(self, num_classes, in_chans, steps_per_epoch, learning_rate, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_chans
        )
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.logger.experiment["optimizer/lr"].log(self.learning_rate)
        optimizer_name = optimizer.__class__.__name__
        self.logger.experiment["optimizer/name"].log(optimizer_name)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.max_epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_f1"}
