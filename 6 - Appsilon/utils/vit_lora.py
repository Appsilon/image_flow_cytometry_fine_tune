import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageClassification, AutoImageProcessor


class VitModel(LightningModule):
    def __init__(self, num_classes, in_chans, steps_per_epoch, learning_rate, max_epochs, 
                 lora_r_alpha, lora_target_modules, lora_dropout, lora_bias):
        super().__init__()
        self.save_hyperparameters()
        non_lora_model = AutoModelForImageClassification.from_pretrained(
                        "google/vit-base-patch16-224-in21k",
                        ignore_mismatched_sizes=True,
                        num_labels = num_classes,
                        num_channels=in_chans,
)
        # self.model_name = non_lora_model.__class__.__name__

        print("> Initializing models...")
        
        lora_config = LoraConfig(
            r=lora_r_alpha,
            lora_alpha=lora_r_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            use_rslora=True,
            # modules_to_save="classifier"
        )

        self.model = get_peft_model(non_lora_model, lora_config)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch

        print(self.model.print_trainable_parameters())

        self.train_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1_score = F1Score(num_classes=num_classes, task="multiclass", average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc(logits, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, labels), prog_bar=True)
        self.log("val_f1", self.f1_score(logits, labels), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.logger.experiment["optimizer/lr"].log(self.learning_rate)
        optimizer_name = optimizer.__class__.__name__
        self.logger.experiment["optimizer/name"].log(optimizer_name)
        # self.logger.experiment["model/name"].log(self.model_name)

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
