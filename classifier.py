from pathlib import Path
import pandas as pd
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F

from utils import get_model, get_loss, get_optimizer, get_scheduler
from models.aggregators.transformer import Transformer
from models.aggregators.attentionmil import AttentionMIL


class ClassifierLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = 514 if config.pos_enc == "ConcatEmbedding" else 512
        self.model = get_model(self.config.model, num_classes=self.config.num_classes, input_dim=config.input_dim, dim=dim, pos_enc=config.pos_enc)
        self.criterion = get_loss(config.criterion, pos_weight=config.pos_weight) if config.task == "binary" else get_loss(config.criterion)
        # TODO save config file correctly (with self.save_hyperparameters?)
        self.save_hyperparameters()
        
        self.lr = config.lr
        self.wd = config.wd

        self.acc_train = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )
        self.acc_val = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )
        self.acc_test = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )

        self.auroc_val = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.f1_test = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.precision_val = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.recall_test = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )

    def forward(self, x, *args):
        logits = self.model(x, *args)
        return logits

    def configure_optimizers(self):
        optimizer = get_optimizer(
            name=self.config.optimizer,
            model=self.model, 
            lr=self.lr,
            wd=self.wd
        )
        scheduler = get_scheduler(
            name=self.config.scheduler,
            optimizer=optimizer
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)
        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float()) 
        else:           
            loss = self.criterion(logits, y)            
        preds = torch.argmax(logits, dim=1, keepdim=True)
        # self.lr_schedulers().step()

        if self.config.task == "binary":
            self.acc_train(preds, y.unsqueeze(1))
        else:
            probs = torch.softmax(logits, dim=1)
            self.acc_train(probs, y)
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)
        # if config.task == "multiclass":
        #     y = y.squeeze(-1)
        loss = self.criterion(logits, y)
        # probs = torch.sigmoid(logits)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1, keepdim=True)
        
        self.acc_val(probs, y)  # preds
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)

        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision/val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall/val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity/val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True)
    
    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, coords, y, _, patient = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)
        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float()) 
        else:           
            loss = self.criterion(logits, y)      
        # probs = torch.sigmoid(logits)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1, keepdim=True)
        
        if self.config.task == "binary":
            y = y.unsqueeze(1)
        self.acc_test(probs, y)  # preds
        self.auroc_test(probs, y)
        self.f1_test(probs, y)
        self.precision_test(probs, y)
        self.recall_test(probs, y)
        self.specificity_test(probs, y)

        self.log("loss/test", loss, prog_bar=False)
        self.log("acc/test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision/test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall/test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity/test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True)

        # TODO rewrite for batch size > 1 (not needed atm bc bs=1 always in testing mode)
        outputs = pd.DataFrame(
            data=[[patient[0], y.item(), preds.item(), logits.squeeze(), (y==preds).int().item()]], 
            columns=['patient', 'ground_truth', 'prediction', 'logits', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)
