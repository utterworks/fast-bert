import os
import torch
from .data_cls import BertDataBunch
from .data_ner import BertNERDataBunch
from .learner_cls import BertLearner
from .learner_ner import BertNERLearner

from transformers import AutoTokenizer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class BertClassificationPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        multi_label=False,
        model_type="bert",
        use_fast_tokenizer=True,
        do_lower_case=True,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_path = model_path
        self.label_path = label_path
        self.multi_label = multi_label
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.device = device

        # Use auto-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=use_fast_tokenizer
        )

        self.learner = self.get_learner()

    def get_learner(self):
        databunch = BertDataBunch(
            self.label_path,
            self.label_path,
            self.tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=512,
            multi_gpu=False,
            multi_label=self.multi_label,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = BertLearner.from_pretrained_model(
            databunch,
            self.model_path,
            metrics=[],
            device=self.device,
            logger=None,
            output_dir=None,
            warmup_steps=0,
            multi_gpu=False,
            is_fp16=False,
            multi_label=self.multi_label,
            logging_steps=0,
        )

        return learner

    def predict_batch(self, texts):
        return self.learner.predict_batch(texts)

    def predict(self, text):
        predictions = self.predict_batch([text])[0]
        return predictions


class BertNERPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        model_type="bert",
        use_fast_tokenizer=True,
        do_lower_case=True,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_path = model_path
        self.label_path = label_path
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.device = device

        # Use auto-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=use_fast_tokenizer
        )

        self.learner = self.get_learner()

    def get_learner(self):
        databunch = BertNERDataBunch(
            self.label_path,
            self.tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=512,
            multi_gpu=False,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = BertNERLearner.from_pretrained_model(
            databunch,
            self.model_path,
            device=self.device,
            logger=None,
            output_dir=None,
            warmup_steps=0,
            multi_gpu=False,
            is_fp16=False,
            logging_steps=0,
        )

        return learner

    def predict_batch(self, texts, group=True, exclude_entities=["O"]):
        predictions = []

        for text in texts:
            pred = self.predict(text, group=group, exclude_entities=exclude_entities)
            if pred:
                predictions.append({"text": text, "results": pred})

    def predict(self, text, group=True, exclude_entities=["O"]):
        predictions = self.learner.predict(
            text, group=group, exclude_entities=exclude_entities
        )
        return predictions
