# -*- coding: utf-8 -*-
r""" 
Text Tokenizer
==============
    Wrapper around GPT2 tokenizer.
"""
import torch
from torchnlp.encoders.text.text_encoder import TextEncoder
from transformers import AutoTokenizer


# Tokens used to anonymize names and religions
SPECIAL_TOKENS = ["[NAME]", "[RELIGION]"]


class Tokenizer(TextEncoder):
    """Wrapper around Hugging-face Auto-tokenizer.

    :param pretrained_model: Transformer pretrained model.
    """

    def __init__(self, pretrained_model) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        orig_vocab = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.vocab_size = orig_vocab + num_added_tokens

        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.eos_token_id
        self.vocab = self.tokenizer.get_vocab()

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        :return: torch.Tensor with Encoding of the `sequence`.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.tokenizer(sequence, truncation=True)["input_ids"]