import os
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .data import BertDataBunch
from .learner import BertLearner

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class BertClassificationPredictor(object):

    def __init__(self, model_path, pretrained_path, label_path, multi_label=False):
        self.model_path = model_path
        self.pretrained_path = pretrained_path
        self.label_path = label_path
        self.multi_label = multi_label

        self.learner = self.get_learner()

    def get_learner(self):
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_path, do_lower_case=True)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        databunch = BertDataBunch(self.label_path, self.label_path, tokenizer, train_file=None, val_file=None,
                                  bs=32, maxlen=512, multi_gpu=False, multi_label=self.multi_label)

        learner = BertLearner.from_pretrained_model(databunch, self.pretrained_path, [], device, None,
                                                    finetuned_wgts_path=self.model_path,
                                                    is_fp16=False, loss_scale=128, multi_label=self.multi_label)
        return learner

    def predict_batch(self, texts):
        return self.learner.predict_batch(texts)

    def predict(self, text):
        predictions = self.predict_batch([text])[0]
        return predictions
