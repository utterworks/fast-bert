import os
import copy
import logging
from packaging import version
from .data_cls import BertDataBunch, InputExample, InputFeatures
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from .learner_util import Learner
from torch import nn
from typing import List

from .modeling import (
    BertForMultiLabelSequenceClassification,
    XLNetForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification,
    DistilBertForMultiLabelSequenceClassification,
    CamembertForMultiLabelSequenceClassification,
    AlbertForMultiLabelSequenceClassification,
    ElectraForMultiLabelSequenceClassification,
    FlaubertForMultiLabelSequenceClassification
)

from .bert_layers import BertLayerNorm
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from pathlib import Path

from torch.optim.lr_scheduler import _LRScheduler, Optimizer

from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
)

from transformers import AutoModelForSequenceClassification, AutoConfig

PYTORCH_VERSION = version.parse(torch.__version__)

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        (BertForSequenceClassification, BertForMultiLabelSequenceClassification),
        BertTokenizer,
    ),
    "xlnet": (
        XLNetConfig,
        (XLNetForSequenceClassification, XLNetForMultiLabelSequenceClassification),
        XLNetTokenizer,
    ),
    "xlm": (
        XLMConfig,
        (XLMForSequenceClassification, XLMForSequenceClassification),
        XLMTokenizer,
    ),
    "roberta": (
        RobertaConfig,
        (RobertaForSequenceClassification, RobertaForMultiLabelSequenceClassification),
        RobertaTokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        (
            DistilBertForSequenceClassification,
            DistilBertForMultiLabelSequenceClassification,
        ),
        DistilBertTokenizer,
    ),
    "albert": (
        AlbertConfig,
        (AlbertForSequenceClassification, AlbertForMultiLabelSequenceClassification),
        AlbertTokenizer,
    ),
    "camembert-base": (
        CamembertConfig,
        (
            CamembertForSequenceClassification,
            CamembertForMultiLabelSequenceClassification,
        ),
        CamembertTokenizer,
    ),
    "electra": (
        ElectraConfig,
        (ElectraForSequenceClassification, ElectraForMultiLabelSequenceClassification),
        ElectraTokenizer,
    ),
    "flaubert": (    
        FlaubertConfig,      
        (FlaubertForSequenceClassification, FlaubertForMultiLabelSequenceClassification),
        FlaubertTokenizer,
    ),
}
if version.parse(torch.__version__) >= version.parse("1.6"):
    IS_AMP_AVAILABLE = True
    from torch.cuda.amp import autocast
else:
    IS_AMP_AVAILABLE = False


def load_model(
    dataBunch,
    pretrained_path,
    finetuned_wgts_path,
    device,
    multi_label,
    pos_weight,
    weight,
):

    model_type = dataBunch.model_type
    model_state_dict = None

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    if finetuned_wgts_path:
        model_state_dict = torch.load(finetuned_wgts_path, map_location=map_location)
    else:
        model_state_dict = None

    if multi_label is True:
        config_class, model_class, _ = MODEL_CLASSES[model_type]

        model_class[1].pos_weight = pos_weight if pos_weight is not None else dataBunch.pos_weight
        model_class[1].weight = weight if weight is not None else dataBunch.weight

        config = config_class.from_pretrained(
            str(pretrained_path), num_labels=len(dataBunch.labels)
        )

        model = model_class[1].from_pretrained(
            str(pretrained_path), config=config, state_dict=model_state_dict
        )
    else:
        if finetuned_wgts_path:
            finetuned_path = str(finetuned_wgts_path).replace("/pytorch_model.bin", "")
            config = AutoConfig.from_pretrained(
                str(finetuned_path), num_labels=len(dataBunch.labels)
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                str(finetuned_path), config=config
            )
        else:
            config = AutoConfig.from_pretrained(
                str(pretrained_path), num_labels=len(dataBunch.labels)
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(pretrained_path), config=config
            )

    return model.to(device)


class BertLearner(Learner):
    @staticmethod
    def from_pretrained_model(
        dataBunch,
        pretrained_path,
        output_dir,
        metrics,
        device,
        logger=logging.getLogger(__name__),
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        freeze_transformer_layers=False,
        pos_weight=None,
        weight=None,
    ):
        if is_fp16 and (IS_AMP_AVAILABLE is False):
            logger.debug("Apex not installed. switching off FP16 training")
            is_fp16 = False

        model = load_model(
            dataBunch,
            pretrained_path,
            finetuned_wgts_path,
            device,
            multi_label,
            pos_weight,
            weight,
        )

        return BertLearner(
            dataBunch,
            model,
            str(pretrained_path),
            output_dir,
            metrics,
            device,
            logger,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            multi_label,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
            freeze_transformer_layers,
        )

    def __init__(
        self,
        data: BertDataBunch,
        model: nn.Module,
        pretrained_model_path,
        output_dir,
        metrics,
        device,
        logger,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        freeze_transformer_layers=False,
    ):

        super(BertLearner, self).__init__(
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

        # Classification specific attributes
        self.multi_label = multi_label
        self.metrics = metrics

        # LR Finder
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.state_cacher = None

        self.scaler = torch.cuda.amp.GradScaler() if is_fp16 is True else None

        # Freezing transformer model layers
        if freeze_transformer_layers:
            for name, param in self.model.named_parameters():
                if name.startswith(data.model_type):
                    param.requires_grad = False

    ### Train the model ###
    def fit(
        self,
        epochs,
        lr,
        validate=True,
        return_results=False,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    ):
        results_val = []
        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = (
                self.max_steps // len(train_dataloader) // self.grad_accumulation_steps
                + 1
            )
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs

        # Prepare optimiser
        optimizer = self.get_optimizer(lr, optimizer_type=optimizer_type)

        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, "module"):
            self.model = self.model.module

        # Get scheduler
        scheduler = self.get_scheduler(
            optimizer, t_total=t_total, schedule_type=schedule_type
        )

        # Parallelize the model architecture
        if self.multi_gpu is True:
            self.model = torch.nn.DataParallel(self.model)

        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.data.train_batch_size * self.grad_accumulation_steps,
        )
        self.logger.info(
            "  Gradient Accumulation steps = %d", self.grad_accumulation_steps
        )
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))

        for epoch in pbar:
            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                # Run training step and get loss
                loss = self.training_step(batch)

                tr_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if self.is_fp16:
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(optimizer)

                    if self.is_fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                                self.logger.info(
                                    "eval_{} after step {}: {}: ".format(
                                        key, global_step, value
                                    )
                                )

                        # Log metrics
                        self.logger.info(
                            "lr after step {}: {}".format(
                                global_step, scheduler.get_lr()[0]
                            )
                        )
                        self.logger.info(
                            "train_loss after step {}: {}".format(
                                global_step,
                                (tr_loss - logging_loss) / self.logging_steps,
                            )
                        )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.logging_steps,
                            global_step,
                        )

                        logging_loss = tr_loss

            # Evaluate the model against validation set after every epoch
            if validate:
                results = self.validate()
                for key, value in results.items():
                    self.logger.info(
                        "eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value)
                    )
                results_val.append(results)

            # Log metrics
            self.logger.info(
                "lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0])
            )
            self.logger.info(
                "train_loss after epoch {}: {}".format(
                    (epoch + 1), epoch_loss / epoch_step
                )
            )
            self.logger.info("\n")

        tb_writer.close()

        if return_results:
            return global_step, tr_loss / global_step, results_val
        else:
            return global_step, tr_loss / global_step

    ### Training Step
    def training_step(self, batch):
        self.model.train()
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }

        if self.model_type in ["bert", "xlnet"]:
            inputs["token_type_ids"] = batch[2]

        if self.is_fp16:
            with autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        loss = outputs[0]

        if self.n_gpu > 1:
            loss = loss.mean()

        if self.grad_accumulation_steps > 1:
            loss = loss / self.grad_accumulation_steps

        if self.is_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    ### Evaluate the model
    def validate(self, quiet=False, loss_only=False, return_preds=False):
        if quiet is False:
            self.logger.info("Running evaluation")
            self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
            self.logger.info("  Batch size = %d", self.data.val_batch_size)

        all_logits = None
        all_labels = None

        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        preds = None
        out_label_ids = None

        validation_scores = {metric["name"]: 0.0 for metric in self.metrics}

        iterator = self.data.val_dl if quiet else progress_bar(self.data.val_dl)

        for step, batch in enumerate(iterator):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            nb_eval_examples += inputs["input_ids"].size(0)

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = inputs["labels"]
            else:
                all_labels = torch.cat((all_labels, inputs["labels"]), 0)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps

        results = {"loss": eval_loss}

        if return_preds:
            results["y_preds"] = np.argmax(all_logits.detach().cpu().numpy(), axis=1)
            results["y_true"] = all_labels.detach().cpu().numpy()

        if loss_only is False:
            # Evaluation metrics
            for metric in self.metrics:
                validation_scores[metric["name"]] = metric["function"](
                    all_logits, all_labels, labels=self.data.labels
                )
            results.update(validation_scores)

        return results

    ### Return Predictions ###
    def predict_batch(self, texts=None, verbose=True):

        if verbose:
            if self.logger is None:
                self.logger = logging.getLogger(__name__)
        if texts:
            if verbose:
                self.logger.info("---PROGRESS-STATUS---: Tokenizing input texts...")
            dl = self.data.get_dl_from_texts(texts)
            if verbose:
                self.logger.info("---PROGRESS-STATUS---: Tokenizing input texts...DONE")
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl

        all_logits = None

        self.model.eval()
        for step, batch in enumerate(dl):
            if verbose:
                self.logger.info(
                    "---PROGRESS-STATUS---: Predicting batch {}/{}".format(
                        step + 1, len(dl)
                    )
                )
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                if self.multi_label:
                    logits = logits.sigmoid()
                # elif len(self.data.labels) == 2:
                #     logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

        result_df = pd.DataFrame(all_logits, columns=self.data.labels)
        results = result_df.to_dict(orient="records")

        if verbose:
            self.logger.info("---PROGRESS-STATUS---: Predicting batch...DONE")
        return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]

    # Begin code for LR Finder
    # Courtesy https://github.com/davidtvs/pytorch-lr-finder

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        if hasattr(self.model, "module"):
            self.model = self.model.module
        self.model.load_state_dict(self.state_cacher.retrieve("model"))
        self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        self.model.to(self.device)

    def lr_find(
        self,
        start_lr,
        end_lr=10,
        use_val_loss=True,
        optimizer_type="lamb",
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.
        Arguments:
            start_lr (float, optional): the starting learning rate for the range test.
                Default: None (uses the learning rate from the optimizer).
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        Reference:
        [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](
        https://medium.com/huggingface/ec88c3e51255)
        [thomwolf/gradient_accumulation](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3)
        """

        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.state_cacher = StateCacher(True, cache_dir=self.output_dir)

        self.optimizer = self.get_optimizer(lr=start_lr, optimizer_type=optimizer_type)

        if hasattr(self.model, "module"):
            self.model = self.model.module

        self.state_cacher.store("model", self.model.state_dict())
        self.state_cacher.store("optimizer", self.optimizer.state_dict())

        # Parallelize the model architecture
        if self.multi_gpu is True:
            self.model = torch.nn.DataParallel(self.model)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        train_iter = TrainDataLoaderIter(self.data.train_dl)

        for iteration in tqdm(range(num_iter)):
            # train on batch and retrieve loss
            loss = self._train_batch(train_iter)
            if use_val_loss:
                loss = self.validate(quiet=True, loss_only=True)["loss"]

            # Update the learning rate
            self.history["lr"].append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")
        self.reset()
        self.plot()

    def _train_batch(self, train_iter):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for i in range(self.grad_accumulation_steps):
            batch = next(train_iter)

            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            if self.is_fp16:
                with autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            loss = outputs[0]

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss /= self.grad_accumulation_steps

            if self.is_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        self.optimizer.step()

        return total_loss.item()

    def _validate(self, val_iter):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in val_iter:
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                batch_size = batch[0].size(0)

                loss = self.model(**inputs)[0]

                running_loss += loss.item() * batch_size

        return running_loss / len(val_iter.dataset)

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
        Returns:
            The matplotlib.axes.Axes object that contains the plot.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    "Failed to load state in {}. File doesn't exist anymore.".format(fn)
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])


class DataLoaderIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        if not isinstance(batch_data, list) and not isinstance(batch_data, tuple):
            raise ValueError(
                "Your batch type not supported: {}. Please inherit from "
                "`TrainDataLoaderIter` (or `ValDataLoaderIter`) and redefine "
                "`_batch_make_inputs_labels` method.".format(type(batch_data))
            )

        inputs, labels, *_ = batch_data

        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return batch
        # return self.inputs_labels_from_batch(batch)


class TrainDataLoaderIter(DataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            # inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            # inputs, labels = self.inputs_labels_from_batch(batch)

        return batch


class ValDataLoaderIter(TrainDataLoaderIter):
    pass
