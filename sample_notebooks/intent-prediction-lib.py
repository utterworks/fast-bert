# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'sample_notebooks'))
    print(os.getcwd())
except:
    pass

# %%
import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from energy_bert.data import BertDataBunch
from energy_bert.learner import BertLearner
from energy_bert.utils.spellcheck import BingSpellCheck
from pathlib import Path


# %%
# !pip --no-cache-dir install git+https://e791691795db788356f2d576c50aa90829425c7e@github.com/kaushaltrivedi/energy-bert.git --upgrade


# %%
MODEL_PATH = Path(
    '../models/intent_classification_lib_2019-02-17_23-26-25.bin')
BERT_PRETRAINED_PATH = Path(
    '../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
LABEL_PATH = Path('../labels')


# %%
tokenizer = BertTokenizer.from_pretrained(
    BERT_PRETRAINED_PATH, do_lower_case=True)


# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# %%
device


# %%
databunch = BertDataBunch(LABEL_PATH, LABEL_PATH, tokenizer, train_file=None, val_file=None,
                          bs=32, maxlen=512, multi_gpu=False, multi_label=False)


# %%
num_labels = len(databunch.labels)


# %%
learner = BertLearner.from_pretrained_model(databunch, BERT_PRETRAINED_PATH, [], device, None,
                                            finetuned_wgts_path=MODEL_PATH,
                                            is_fp16=False, loss_scale=128, multi_label=False)


# %%
spellcheck = BingSpellCheck('8cf9679697c9464c881be6c2350f4bbd')


# %%
# %%timeit
text = """
please log a compliant
"""
text = spellcheck.spell_check(text)
learner.predict_batch([text])


# %%
# text = spellcheck.spell_check('the worl is not enuf')
text


# %%
