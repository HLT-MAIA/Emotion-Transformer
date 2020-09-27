# -*- coding: utf-8 -*-
r""" 
EmotionTransformer Model
==================
    Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple
import click

import torch
import yaml
from transformers import AdamW, AutoModel

import torch.nn as nn
import pytorch_lightning as pl
from model.tokenizer import Tokenizer
from pytorch_lightning.metrics.functional import accuracy
from utils import Config
from torchnlp.utils import lengths_to_mask
from model.utils import mask_fill
from torchnlp.utils import collate_tensors

from sklearn.metrics import accuracy_score, f1_score

EKMAN = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]

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
    "neutral"
]


class EmotionTransformer(pl.LightningModule):
    """Hugging-face Transformer Model implementing the PyTorch Lightning interface that 
    can be used to train an Emotion Classifier.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        :param pretrained_model: Pretrained Transformer model to be used.
        :param learning_rate: Learning Rate used during training.
        :param dataset_path: Path to a json file containing our data.
        :param batch_size: Batch Size used during training.
        """

        pretrained_model: str = "roberta-base"

        # Optimizer
        nr_frozen_epochs: int = 1
        encoder_learning_rate: float = 3.25e-5
        learning_rate: float = 6.25e-5
        layerwise_decay: float = 0.95

        # Data configs
        dataset_path: str = ""
        labels: str = "ekman"

        # Training details
        batch_size: int = 2

        
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.transformer = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        
        self.encoder_features = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers + 1

        if self.hparams.labels == "ekman":
            self.label_encoder = {EKMAN[i]: i for i in range(len(EKMAN))}
        elif self.hparams.labels == "goemotions":
            self.label_encoder = {GOEMOTIONS[i]: i for i in range(len(GOEMOTIONS))}
        else:
            raise Exception("unrecognized label set: {}".format(self.hparams.labels))
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, len(self.label_encoder)),
        )

        self.loss = nn.BCEWithLogitsLoss()
        
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs


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

    def layerwise_lr(self, lr: float, decay: float):
        """
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


    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        head_parameters = [
            {"params": self.classification_head.parameters(), "lr": self.hparams.learning_rate}
        ]

        optimizer = AdamW(
            layer_parameters + head_parameters, lr=self.hparams.learning_rate, correct_bias=True
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
        
        # Average Pooling
        word_embeddings = mask_fill(
            0.0, input_ids, word_embeddings, self.tokenizer.pad_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        # Classify
        return self.classification_head(sentemb)


    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        input_ids, input_lengths, labels = batch
        logits = self.forward(input_ids, input_lengths)
        loss_value = self.loss(logits, labels)

        # Train Metrics
        tqdm_dict = {"train_loss": loss_value}

        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss_value, "progress_bar": tqdm_dict, "log": tqdm_dict}


    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.

        :returns: dictionary passed to the validation_end function.
        """
        input_ids, input_lengths, labels = batch
        logits = self.forward(input_ids, input_lengths)
        loss_value = self.loss(logits, labels)
        
        # Turn logits into probabilities
        y_pred = torch.sigmoid(logits)

        # Turn probabilities into binary predictions
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred.cpu().numpy()
        y = labels.cpu().numpy()
        
        # Return metrics
        return {
            "val_loss": loss_value.cpu().numpy(),
            "val_accuracy": accuracy_score(y_pred, y),
            "val_macro-F1": f1_score(y_pred, y, average='macro', zero_division=0),
            #"val_hamming": hamming_loss(y_pred, y)
        }


    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        # convert List of Dicts to Dict of lists
        outputs = collate_tensors(outputs)
        # Average results for all validation batches.
        outputs = {metric: torch.tensor(sum(values)/len(values)) for metric, values in outputs.items()}
        return {
            "progress_bar": outputs,
            "log": outputs,
            "val_loss": outputs["val_loss"],
        }

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return:Pretrained model.
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