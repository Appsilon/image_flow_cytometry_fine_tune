import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from timm import create_model
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ConvnextModel(LightningModule):
    def __init__(self, num_classes, in_chans, steps_per_epoch, learning_rate, max_epochs, weight_decay=0, dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate = dropout
        )

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.weight_decay = weight_decay

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

        # Compute loss
        loss = nn.CrossEntropyLoss()(outputs, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.val_acc(outputs, labels), prog_bar=True)  # Reuse validation accuracy metric
        self.log("test_f1", self.f1_score(outputs, labels), prog_bar=True)  # Reuse F1 score

        preds = torch.argmax(outputs, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())

        return loss

    def on_test_epoch_start(self):
        # Initialize storage for predictions and labels
        self.test_preds = []
        self.test_labels = []
    
    def plot_confusion_matrix(self, cm, class_names):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return fig

    def on_test_epoch_end(self):
        preds = np.array(self.test_preds)
        labels = np.array(self.test_labels)

        conf_matrix = confusion_matrix(labels, preds)
        self.logger.experiment["test/confusion_matrix"].log(conf_matrix.tolist())
        # Plot and log as an image
        class_names = [str(i) for i in range(self.hparams.num_classes)]  # Adjust as needed
        cm_plot = self.plot_confusion_matrix(conf_matrix, class_names)
        self.logger.experiment["test/confusion_matrix_image"].upload(cm_plot)
        
        class_report = classification_report(labels, preds, output_dict=True)
        self.logger.experiment["test/classification_report"].log(class_report)

        self.logger.experiment["test/accuracy"].log(self.val_acc.compute())
        self.logger.experiment["test/f1_score"].log(self.f1_score.compute())

        print("Confusion Matrix:\n", conf_matrix)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.logger.experiment["optimizer/lr"].log(self.learning_rate)
        self.logger.experiment["optimizer/weight_decay"].log(self.weight_decay)
        optimizer_name = optimizer.__class__.__name__
        self.logger.experiment["optimizer/name"].log(optimizer_name)
        self.logger.experiment["model/name"].log(self.model.__class__.__name__)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.max_epochs,
            steps_per_epoch=self.steps_per_epoch
        )

        lr_scheduler_config = {
        'scheduler': scheduler,
        'interval': 'step',  # Step after every batch
        'monitor': 'val_f1'
    }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

