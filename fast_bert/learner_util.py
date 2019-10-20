from pathlib import Path

import torch
from transformers import (ConstantLRSchedule,
                          WarmupCosineSchedule,
                          WarmupConstantSchedule,
                          WarmupLinearSchedule,
                          WarmupCosineWithHardRestartsSchedule,
                          AdamW
                         )
from pytorch_lamb import Lamb

class Learner(object):

    def __init__(self, data, model, pretrained_model_path, output_dir, device, logger,
                 multi_gpu=True, is_fp16=True, warmup_steps=0, fp16_opt_level='O1',
                 grad_accumulation_steps=1, max_grad_norm=1.0, adam_epsilon=1e-8, logging_steps=100):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.max_steps = -1
        self.weight_decay = 0.0
        self.model_type = data.model_type

        self.output_dir = output_dir

        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()


    # Get the optimiser object
    def get_optimizer(self, lr, optimizer_type='lamb'):

        # Prepare optimiser and schedule
        no_decay = {'bias', 'LayerNorm.weight'}

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        if optimizer_type == 'lamb':
            optimizer = Lamb(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        elif optimizer_type == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        elif:
            raise Exception('Unknown optimizer type: %s' % optimizer_type)

        return optimizer

    # Get learning rate scheduler
    def get_scheduler(self, optimizer, t_total, schedule_type='warmup_linear'):

        SCHEDULES = {
            None: ConstantLRSchedule,
            'none': ConstantLRSchedule,
            'warmup_cosine': WarmupCosineSchedule,
            'warmup_constant': WarmupConstantSchedule,
            'warmup_linear': WarmupLinearSchedule,
            'warmup_cosine_hard_restarts': WarmupCosineWithHardRestartsSchedule
        }

        return SCHEDULES[schedule_type](optimizer, warmup_steps=self.warmup_steps, t_total=t_total)

    def save_model(self):
        """ Persist the model to disk.

        :return void:
        """
        path = self.output_dir/'model_out'
        path.mkdir(exist_ok=True)

        torch.cuda.empty_cache()
        # Save a trained model. Only save the model itself.
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)

        # Save the tokenizer.
        self.data.tokenizer.save_pretrained(path)
