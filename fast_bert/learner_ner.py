import os
import torch
import numpy as np
from .data_ner import BertNERDataBunch
from torch import nn
from seqeval.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
from .learner_util import Learner

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
)

try:
    from apex import amp

    IS_AMP_AVAILABLE = True
except ImportError:
    IS_AMP_AVAILABLE = False


def load_model(databunch, pretrained_path, finetuned_wgts_path, device):

    model_type = databunch.model_type
    model_state_dict = None

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    if finetuned_wgts_path:
        model_state_dict = torch.load(finetuned_wgts_path, map_location=map_location)
    else:
        model_state_dict = None

    config = AutoConfig.from_pretrained(
        str(pretrained_path),
        num_labels=len(databunch.labels),
        model_type=model_type,
        id2label=databunch.label_map,
        label2id={label: i for i, label in enumerate(databunch.labels)},
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(pretrained_path), config=config, state_dict=model_state_dict
    )

    return model


class BertNERLearner(Learner):
    @staticmethod
    def from_pretrained_model(
        databunch,
        pretrained_path,
        output_dir,
        device,
        logger,
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
    ):
        if is_fp16 and (IS_AMP_AVAILABLE is False):
            logger.debug("Apex not installed. switching off FP16 training")
            is_fp16 = False

        model = load_model(databunch, pretrained_path, finetuned_wgts_path, device)

        return BertNERLearner(
            databunch,
            model,
            str(pretrained_path),
            output_dir,
            device,
            logger,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

    def __init__(
        self,
        data: BertNERDataBunch,
        model: nn.Module,
        pretrained_model_path,
        output_dir,
        device,
        logger,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
    ):

        super(BertNERLearner, self).__init__(
            data,
            model,
            pretrained_model_path,
            output_dir,
            device,
            logger,
            multi_gpu,
            is_fp16,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        self.training_arguments = TrainingArguments(
            str(output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=data.batch_size_per_gpu,
            per_device_eval_batch_size=data.batch_size_per_gpu * 2,
            gradient_accumulation_steps=grad_accumulation_steps,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            logging_dir=str(tensorboard_dir),
            logging_steps=logging_steps,
            fp16=is_fp16,
            fp16_opt_level=fp16_opt_level,
        )

        # LR Finder
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.state_cacher = None

    def fit(self, epochs, lr, validate=True):

        self.training_arguments.learning_rate = lr
        self.training_arguments.num_train_epochs = epochs
        self.training_arguments.do_eval = validate

        self.get_trainer().train()

    def validate(self, quiet=False, loss_only=False):
        if quiet is False:
            self.logger.info("Running evaluation")
            self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
            self.logger.info("  Batch size = %d", self.data.val_batch_size)

        result = self.get_trainer().evaluate()
        return result

    def predict(self, text):
        label_list = self.data.labels

        tokenizer = self.data.tokenizer
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        inputs = tokenizer.encode(text, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        return [
            (token, label_list[prediction])
            for token, prediction in zip(tokens, predictions[0].tolist())
        ]

    def save_model(self, path=None):

        if not path:
            path = self.output_dir / "model_out"

        path.mkdir(exist_ok=True)

        # Convert path to str for save_pretrained calls
        path = str(path)

        torch.cuda.empty_cache()
        # Save a trained model
        self.get_trainer().save_model(path)

        # save the tokenizer
        if self.get_trainer().is_world_master():
            self.data.tokenizer.save_pretrained(path)

    def get_trainer(self):
        if self.trainer is None:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_arguments,
                train_dataset=self.data.train_dl.dataset,
                eval_dataset=self.data.val_dl.dataset,
                compute_metrics=self.compute_metrics,
            )

        return self.trainer

    def align_predictions(
        self, predictions: np.ndarray, label_ids: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.data.label_map[label_ids[i][j]])
                    preds_list[i].append(self.data.label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        preds_list, out_label_list = self.align_predictions(p.predictions, p.label_ids)
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
