import os
from .data import BertDataBunch, InputExample, InputFeatures
from .modeling import BertForMultiLabelSequenceClassification
from torch.optim.lr_scheduler import _LRScheduler, Optimizer
from pytorch_transformers import AdamW, ConstantLRSchedule, WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from pytorch_transformers import BertForSequenceClassification
from .bert_layers import BertLayerNorm
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
    from .bert_layers import BertLayerNorm as FusedLayerNorm
    

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    None:       ConstantLRSchedule,
    "none":     ConstantLRSchedule,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule,
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
                              multi_gpu=True, is_fp16=True, loss_scale=0, warmup_proportion=0.1, fp16_opt_level='O1',
                              grad_accumulation_steps=1, multi_label=False, max_grad_norm=1.0):
        
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
        
        device_id = torch.cuda.current_device()
        
        model.to(device)
        
#        if device.type == 'cuda':
#            if multi_gpu == False:
#                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id],
#                                                                  output_device=device_id,
#                                                                  find_unused_parameters=True)
#            else:
#                model = torch.nn.DataParallel(model)
            
        return BertLearner(dataBunch, model, pretrained_path, metrics, device, logger, 
                           multi_gpu, is_fp16, loss_scale, warmup_proportion, fp16_opt_level, grad_accumulation_steps, multi_label, max_grad_norm )
            
        
        
        
    def __init__(self, data: BertDataBunch, model: nn.Module, pretrained_model_path, metrics, device,logger,
                 multi_gpu=True, is_fp16=True, loss_scale=0, warmup_proportion=0.1, fp16_opt_level='O1',
                 grad_accumulation_steps=1, multi_label=False, max_grad_norm=1.0):
        
        self.multi_label = multi_label
        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.metrics = metrics
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.loss_scale = loss_scale
        self.warmup_proportion = warmup_proportion
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.bn_types = (BertLayerNorm, FusedLayerNorm)
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        
        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        
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
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr) 
        
        warmup_steps = self.warmup_proportion * t_total
        schedule_class = SCHEDULES[schedule_type]
        schedule = schedule_class(optimizer, warmup_steps=warmup_steps, t_total=t_total)
        
        if self.is_fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.fp16_opt_level)
        
        return optimizer, schedule
    
    def validate(self):
        self.logger.info("Running evaluation")
        
        self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
        self.logger.info("  Batch size = %d", self.data.bs)
        
        all_logits = None
        all_labels = None
        
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        preds = None
        out_label_ids = None
        
        
        validation_scores = {metric['name']: 0. for metric in self.metrics}
        validation_scores2 = {metric['name']: 0. for metric in self.metrics}
        
        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2], 
                          'labels':         batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                
                eval_loss += tmp_eval_loss.mean().item()
                
            # tmp_eval_accuracy = self.metrics[0]['function'](logits, inputs['labels'])
            
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = inputs['labels']
            else:   
                all_labels =  torch.cat((all_labels, inputs['labels']), 0)
            
            nb_eval_examples += inputs['input_ids'].size(0)
            
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        
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
    
    def validate_old(self):
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
    
#        if self.multi_gpu == False:
#            try:
#                from apex.parallel import DistributedDataParallel as DDP
#            except ImportError:
#                raise ImportError("Please install apex distributed and fp16 training.")
#
#            self.model = DDP(self.model)
#        else:
#            self.model = torch.nn.DataParallel(self.model)
    
    def fit(self, epochs, lr, validate=True, schedule_type="warmup_linear"):
        
        num_train_steps = int(len(self.data.train_dl) / self.grad_accumulation_steps * epochs)
        
        if self.optimizer is None:
            self.optimizer, self.schedule = self.get_optimizer(lr , num_train_steps)
        
        if self.is_fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        # Parallelize the model architecture
        if self.multi_gpu == False:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex distributed and fp16 training.")

            self.model = DDP(self.model)
        else:
            self.model = torch.nn.DataParallel(self.model)
        
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.data.train_dl))
        self.logger.info("  Num Epochs = %d", epochs)
        
        t_total = num_train_steps
        if self.multi_gpu == False:
            t_total = t_total // torch.distributed.get_world_size()
            
        self.logger.info("  Gradient Accumulation steps = %d", self.grad_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)
        
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        
        pbar = master_bar(range(epochs))
        
        for epoch in pbar:
            self.model.train()
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(progress_bar(self.data.train_dl, parent=pbar)):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]}
                
                outputs = self.model(**inputs)
                loss = outputs[0] # model outputs are always tuple in pytorch-transformers (see doc)
                
                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                    
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps
                    
                
                if self.is_fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
            
                tr_loss += loss.item()
                nb_tr_examples += inputs['input_ids'].size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.grad_accumulation_steps == 0:
                    self.schedule.step()  # Update learning rate schedule
                    self.optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
            
            self.logger.info('Loss after epoch {} - {}'.format(epoch, tr_loss / nb_tr_steps))
        
            if validate:
                self.validate()
        
        
        
        
        
    def fit_old(self, epochs, lr, validate=True, schedule_type="warmup_linear"):
        
        num_train_steps = int(len(self.data.train_dl) / self.grad_accumulation_steps * epochs)
        if self.optimizer is None:
            self.optimizer, self.schedule = self.get_optimizer(lr , num_train_steps)
        
        if self.multi_gpu == False:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex distributed and fp16 training.")

            self.model = DDP(self.model)
        else:
            self.model = torch.nn.DataParallel(self.model)
        
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
