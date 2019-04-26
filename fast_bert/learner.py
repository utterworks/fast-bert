import os
from .data import BertDataBunch, InputExample, InputFeatures
from .modeling import BertForMultiLabelSequenceClassification
from torch.optim.lr_scheduler import _LRScheduler, Optimizer
from pytorch_pretrained_bert.optimization import BertAdam, ConstantLR, WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineWithWarmupRestartsSchedule, WarmupCosineWithHardRestartsSchedule
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertLayerNorm
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from fastai.torch_core import *
from fastai.callback import *

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
except:
    from pytorch_pretrained_bert.modeling import BertLayerNorm as FusedLayerNorm

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    None:       ConstantLR,
    "none":     ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule,
    "warmup_cosine_warmpup_restarts": WarmupCosineWithWarmupRestartsSchedule,
    "warmup_cosine_hard_restarts": WarmupCosineWithHardRestartsSchedule
}

class BertLearner(object):
    data:BertDataBunch
    model:torch.nn.Module
#     opt_func
#     loss_func
#     metrics
#     path:str = None
#     model_dir:str = 'models'
    
    @staticmethod
    def from_pretrained_model(dataBunch, pretrained_path, metrics, device, logger, finetuned_wgts_path=None, 
                              multi_gpu=True, is_fp16=True, loss_scale=0, warmup_proportion=0.1, 
                              grad_accumulation_steps=1, multi_label=False):
        
        model_state_dict = None
        
        if finetuned_wgts_path:
            model_state_dict = torch.load(finetuned_wgts_path)
        
        if multi_label == True:
            model = BertForMultiLabelSequenceClassification.from_pretrained(pretrained_path, 
                                                                  num_labels = len(dataBunch.labels), 
                                                                  state_dict=model_state_dict)
        else:
            model = BertForSequenceClassification.from_pretrained(pretrained_path, 
                                                                  num_labels = len(dataBunch.labels), 
                                                                  state_dict=model_state_dict)
                
        if is_fp16:
            model = model.half()
        
        model.to(device)
        
        if device.type == 'cuda':
            if multi_gpu == False:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError("Please install apex to use distributed and fp16 training.")

                model = DDP(model)
            else:
                model = torch.nn.DataParallel(model)
            
        return BertLearner(dataBunch, model, pretrained_path, metrics, device, logger, 
                           multi_gpu, is_fp16, loss_scale, warmup_proportion, grad_accumulation_steps, multi_label )
            
        
        
        
    def __init__(self, data: BertDataBunch, model: nn.Module, pretrained_model_path, metrics, device,logger,
                 multi_gpu=True, is_fp16=True, loss_scale=0, warmup_proportion=0.1, 
                 grad_accumulation_steps=1, multi_label=False):
        
        self.multi_label = multi_label
        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.metrics = metrics
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.loss_scale = loss_scale
        self.warmup_proportion = warmup_proportion
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.bn_types = (BertLayerNorm, FusedLayerNorm)
        
        # split models
        # self.split(self.bert_clas_split)
        #self.layer_groups = self.bert_clas_split()
        
        # self.freeze()
    
    
    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not isinstance(l, self.bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)
        self.optimizer = None
    
#    def freeze_to(self, n:int)->None:
#        for g in self.layer_groups[:n]:
#            for m in g:
#                self.freeze_module(m)
#                
#        for g in self.layer_groups[n:]: 
#            for m in g:
#                self.unfreeze_module(m)
#        
#        self.optimizer = None
                
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
            
    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def freeze(self)->None:
        "Freeze up to last layer group."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)
        self.optimizer = None

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)
        self.optimizer = None
    
    def bert_clas_split(self) -> List[nn.Module]:
        "Split the BERT `model` in groups for differential learning rates."
        if self.model.module:
            model = self.model.module
        else:
            model = self.model
        
        bert = model.bert
        
        embedder = bert.embeddings
        pooler = bert.pooler
        
        encoder = bert.encoder
        
        classifier = [model.dropout, model.classifier]
        
        n = len(encoder.layer)//3
        
        groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n:2*n]), list(encoder.layer[2*n:]), [pooler], classifier]
        return groups
    
    
    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on()
        self.layer_groups = split_model(self.model, split_on)
        return self
    
    
    def get_optimizer(self, lr, num_train_steps, schedule_type='warmup_linear'):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        t_total = num_train_steps
        if self.multi_gpu == False:
            t_total = t_total // torch.distributed.get_world_size()
        
        schedule_class = SCHEDULES[schedule_type]
        schedule = schedule_class(warmup=self.warmup_proportion, t_total=t_total)
        
        if self.is_fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            
            if self.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.loss_scale)
            
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 schedule=schedule,
                                 warmup=self.warmup_proportion,
                                 t_total=t_total)
        
        return optimizer, schedule
    
    def validate(self):
        self.logger.info("Running evaluation")
        
        all_logits = None
        all_labels = None
        
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        validation_scores = {metric['name']: 0. for metric in self.metrics}
        validation_scores2 = {metric['name']: 0. for metric in self.metrics}
        
        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            if self.is_fp16 and self.multi_label:
                label_ids = label_ids.half()
            
            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)

                
                
            tmp_eval_accuracy = self.metrics[0]['function'](logits, label_ids)
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = label_ids
            else:   
                all_labels =  torch.cat((all_labels, label_ids), 0)
            
            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        
        # Evaluation metrics
        for metric in self.metrics:                
            validation_scores[metric['name']] = metric['function'](all_logits, all_labels)
        
        result = {'eval_loss': eval_loss,
                  'metrics': validation_scores }
        
        self.logger.info("Eval results:")
        for key in sorted(result.keys()):
            self.logger.info("  %s = %s", key, str(result[key]))
        
        self.logger.info("--------------------------------------------------------------------------------")
        
        return result
    
    def save_and_reload(self, path, model_name):
        
        torch.cuda.empty_cache() 
        self.model.to('cpu')
        # Save a trained model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        output_model_file = os.path.join(path, "{}.bin".format(model_name))
        torch.save(model_to_save.state_dict(), output_model_file)

        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        if self.multi_label:
            self.model = BertForMultiLabelSequenceClassification.from_pretrained(self.pretrained_model_path, 
                                                                  num_labels = len(self.data.labels), 
                                                                  state_dict=model_state_dict)
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_model_path, 
                                                                  num_labels = len(self.data.labels), 
                                                                  state_dict=model_state_dict)

        if self.is_fp16:
            self.model.half()
        torch.cuda.empty_cache() 
        self.model.to(self.device)
    
        if self.multi_gpu == False:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex distributed and fp16 training.")

            self.model = DDP(self.model)
        else:
            self.model = torch.nn.DataParallel(self.model)
     
    def fit(self, epochs, lr, validate=True, schedule_type="warmup_linear"):
        
        num_train_steps = int(len(self.data.train_dl) / self.grad_accumulation_steps * epochs)
        if self.optimizer is None:
            self.optimizer, self.schedule = self.get_optimizer(lr , num_train_steps)
        
        t_total = num_train_steps
        if self.multi_gpu == False:
            t_total = t_total // torch.distributed.get_world_size()
            
        global_step = 0
        
        pbar = master_bar(range(epochs))
        
        for epoch in pbar:
            self.model.train()
  
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(progress_bar(self.data.train_dl, parent=pbar)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                if self.is_fp16 and self.multi_label:
                    label_ids = label_ids.half()
                
                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                if self.multi_gpu:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps
                
                if self.is_fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.grad_accumulation_steps == 0:
                    if self.is_fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = lr * self.schedule.get_lr(global_step)
                    
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
            self.logger.info('Loss after epoch {} - {}'.format(epoch, tr_loss / nb_tr_steps))
#             logger.info('Eval after epoch  {}'.format(epoch))
        
            if validate:
                self.validate()
    
    def predict_batch(self, texts=None):
        
        if texts:
            dl = self.data.get_dl_from_texts(texts)
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl
            
        all_logits = None

        self.model.eval()

        nb_eval_steps, nb_eval_examples = 0, 0
        for step, batch in enumerate(dl):
            if len(batch) == 4:
                input_ids, input_mask, segment_ids, _ = batch
            else:
                input_ids, input_mask, segment_ids = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                if self.multi_label:
                    logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        result_df =  pd.DataFrame(all_logits, columns=self.data.labels)
        results = result_df.to_dict('record')

        return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]
        
        
        
    

class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

#         if not isinstance(optimizer, Optimizer):
#             raise TypeError('{} is not an Optimizer'.format(
#                 type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

