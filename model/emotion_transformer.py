# -*- coding: utf-8 -*-
r""" 
EmotionTransformer Model
==================
    Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.
"""
import multiprocessing
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import AdamW, AutoModel

from model.data_module import DataModule
from model.tokenizer import Tokenizer
from model.utils import average_pooling, mask_fill, max_pooling
from utils import Config

EKMAN = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

GOEMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class EmotionTransformer(pl.LightningModule):
    """Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        ------------------ Architecture --------------------- 
        :param pretrained_model: Pretrained Transformer model to be used.
        :param pooling: Pooling method for extracting sentence embeddings 
            (options: cls, avg, max, cls+avg)
        
        ----------------- Tranfer Learning --------------------- 
        :param nr_frozen_epochs: number of epochs where the `encoder` model is frozen.
        :param encoder_learning_rate: Learning rate to be used to fine-tune parameters from the `encoder`.
        :param learning_rate: Learning Rate used during training.
        :param layerwise_decay: Learning rate decay for to be applied to the encoder layers.

        ----------------------- Data --------------------- 
        :param dataset_path: Path to a json file containing our data.
        :param labels: Label set (options: `ekman`, `goemotions`)
        :param batch_size: Batch Size used during training.
        """

        pretrained_model: str = "roberta-base"
        pooling: str = "avg"

        # Optimizer
        nr_frozen_epochs: int = 1
        encoder_learning_rate: float = 1.0e-5
        learning_rate: float = 5.0e-5
        layerwise_decay: float = 0.95

        # Data configs
        dataset_path: str = ""
        labels: str = "ekman"

        # Training details
        batch_size: int = 8

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.transformer = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        # Resize embeddings to include the added tokens
        self.transformer.resize_token_embeddings(self.tokenizer.vocab_size)

        self.encoder_features = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers + 1

        if self.hparams.labels == "ekman":
            self.label_encoder = {EKMAN[i]: i for i in range(len(EKMAN))}
        elif self.hparams.labels == "goemotions":
            self.label_encoder = {GOEMOTIONS[i]: i for i in range(len(GOEMOTIONS))}
        else:
            raise Exception("unrecognized label set: {}".format(self.hparams.labels))

        # Classification head
        self.classification_head = nn.Linear(
            self.encoder_features, len(self.label_encoder)
        )
        self.loss = nn.BCEWithLogitsLoss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            click.secho("-- Encoder model fine-tuning", fg="yellow")
            for param in self.transformer.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def layerwise_lr(self, lr: float, decay: float) -> list:
        """ Separates layer parameters and sets the corresponding learning rate to each layer.

        :param lr: Initial Learning rate.
        :param decay: Decay value.

        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        opt_parameters = [
            {
                "params": self.transformer.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        opt_parameters += [
            {
                "params": self.transformer.encoder.layer[l].parameters(),
                "lr": lr * decay ** (self.num_layers - 1 - l),
            }
            for l in range(self.num_layers - 1)
        ]
        return opt_parameters
    
    # Pytorch Lightning Method
    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        head_parameters = [
            {
                "params": self.classification_head.parameters(),
                "lr": self.hparams.learning_rate,
            }
        ]

        optimizer = AdamW(
            layer_parameters + head_parameters,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Reduce unnecessary padding.
        input_ids = input_ids[:, : input_lengths.max()]

        mask = lengths_to_mask(input_lengths, device=input_ids.device)

        # Run model.
        word_embeddings = self.transformer(input_ids, mask)[0]

        # Pooling Layer
        sentemb = self.apply_pooling(input_ids, word_embeddings, mask)

        # Classify
        return self.classification_head(sentemb)

    def apply_pooling(
        self, tokens: torch.Tensor, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """ Gets a sentence embedding by applying a pooling technique to the word-level embeddings.
        
        :param tokens: Tokenized sentences [batch x seq_length].
        :param embeddings: Word embeddings [batch x seq_length x hidden_size].
        :param mask: Mask that indicates padding tokens [batch x seq_length].

        :return: Sentence embeddings [batch x hidden_size].
        """
        if self.hparams.pooling == "max":
            sentemb = max_pooling(tokens, embeddings, self.tokenizer.pad_index)

        elif self.hparams.pooling == "avg":
            sentemb = average_pooling(
                tokens, embeddings, mask, self.tokenizer.pad_index
            )

        elif self.hparams.pooling == "cls":
            sentemb = embeddings[:, 0, :]

        elif self.hparams.pooling == "cls+avg":
            cls_sentemb = embeddings[:, 0, :]
            avg_sentemb = average_pooling(
                tokens, embeddings, mask, self.tokenizer.pad_index
            )
            sentemb = torch.cat((cls_sentemb, avg_sentemb), dim=1)
        else:
            raise Exception("Invalid pooling technique.")

        return sentemb

    # Pytorch Lightning Method
    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels = batch
        logits = self.forward(input_ids, input_lengths)
        loss_value = self.loss(logits, labels)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss_value, "log": {"train_loss": loss_value}}

    # Pytorch Lightning Method
    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels = batch
        logits = self.forward(input_ids, input_lengths)
        loss_value = self.loss(logits, labels)

        # Turn logits into probabilities
        predictions = torch.sigmoid(logits)
        # Turn probabilities into binary predictions
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        return {"val_loss": loss_value, "predictions": predictions, "labels": labels}

    # Pytorch Lightning Method
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)

        # Computes Precision Recall and F1 for all classes
        precision_scores, recall_scores, f1_scores = [], [], []
        for _, index in self.label_encoder.items():
            y_hat = predictions[:, index].cpu().numpy()
            y = labels[:, index].cpu().numpy()
            precision = precision_score(y_hat, y, zero_division=0)
            recall = recall_score(y_hat, y, zero_division=0)
            f1 = (
                0
                if (precision + recall) == 0
                else (2 * (precision * recall) / (precision + recall))
            )
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # We will only log the macro-averaged metrics:
        metrics = {
            "macro-precision": torch.tensor(
                sum(precision_scores) / len(precision_scores)
            ),
            "macro-recall": torch.tensor(sum(recall_scores) / len(recall_scores)),
            "macro-f1": torch.tensor(sum(f1_scores) / len(f1_scores)),
        }
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    # Pytorch Lightning Method
    def test_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ Same as validation_step. """
        return self.validation_step(batch, batch_nb)

    # Pytorch Lightning Method
    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """ Similar to the validation_step_end but computes precision, recall, f1 for each label."""
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        loss_value = torch.stack([o["val_loss"] for o in outputs]).mean()

        # Computes Precision Recall and F1 for all classes
        precision_scores, recall_scores, f1_scores = [], [], []
        for _, index in self.label_encoder.items():
            y_hat = predictions[:, index].cpu().numpy()
            y = labels[:, index].cpu().numpy()
            precision = precision_score(y_hat, y, zero_division=0)
            recall = recall_score(y_hat, y, zero_division=0)
            f1 = (
                0
                if (precision + recall) == 0
                else (2 * (precision * recall) / (precision + recall))
            )
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # We will only log the macro-averaged metrics:
        metrics = {
            "macro-precision": sum(precision_scores) / len(precision_scores),
            "macro-recall": sum(recall_scores) / len(recall_scores),
            "macro-f1": sum(f1_scores) / len(f1_scores),
        }
        for label, i in self.label_encoder.items():
            metrics[label + "-precision"] = precision_scores[i]
            metrics[label + "-recall"] = recall_scores[i]
            metrics[label + "-f1"] = f1_scores[i]

        return {
            "progress_bar": metrics,
            "log": metrics,
            "val_loss": loss_value,
        }

    # Pytorch Lightning Method
    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return: Pretrained model.
        """
        hparams_file = experiment_folder + "hparams.yaml"
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
        ]
        checkpoint_path = experiment_folder + checkpoints[-1]
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model

    def predict(self, samples: List[str]) -> Dict[str, Any]:
        """ Predict function.

        :param samples: list with the texts we want to classify.

        :return: List with classified texts.
        """
        if self.training:
            self.eval()

        output = [{"text": sample} for sample in samples]
        # Create inputs
        input_ids = [self.tokenizer.encode(s) for s in samples]
        input_lengths = [len(ids) for ids in input_ids]
        samples = {"input_ids": input_ids, "input_lengths": input_lengths}
        # Pad inputs
        samples = DataModule.pad_dataset(samples)
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor(samples["input_ids"]),
                torch.tensor(samples["input_lengths"]),
            ),
            batch_size=self.hparams.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

        i = 0
        with torch.no_grad():
            for input_ids, input_lengths in dataloader:
                logits = self.forward(input_ids, input_lengths)
                # Turn logits into probabilities
                probs = torch.sigmoid(logits)
                for j in range(probs.shape[0]):
                    label_probs = {}
                    for label, k in self.label_encoder.items():
                        label_probs[label] = probs[j][k].item()
                    output[i]["emotions"] = label_probs
                    i += 1
        return output
