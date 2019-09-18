import re
import html
import logging
import pandas as pd
import os
import random
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import spacy
from tqdm import tqdm, trange
from fastprogress.fastprogress import master_bar, progress_bar

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

class LMInputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, text_a, text_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class LMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def create_corpus(text_list, target_path, logger=None):

    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'textcat'])

    with open(target_path, 'w') as f:
        #  Split sentences for each document
        logger.info("Formatting corpus for {}".format(target_path))
        for text in progress_bar(text_list):
            
            text = fix_html(text)
            text = replace_multi_newline(text)
            text = spec_add_spaces(text)
            text = rm_useless_spaces(text)
            
            text_lines = [re.sub(r"\n(\s)*","",str(sent)) for i, sent in enumerate(nlp(str(text)).sents)]
            text_lines = [text_line for text_line in text_lines if re.search(r'[a-zA-Z]', text_line)]
            
            f.write('\n'.join(text_lines))
            f.write("\n  \n")

def spec_add_spaces(t:str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r'([/#\n])', r' \1 ', t)

def rm_useless_spaces(t:str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(' {2,}', ' ', t)

def replace_multi_newline(t:str) -> str:
    return re.sub(r"(\n(\s)*){2,}", "\n", t)

def fix_html(x:str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace(' @,@ ',',').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

class LMTextProcessor(object):
    def __init__(self, data_dir, logger=None, encoding='utf-8'):
        self.data_dir = data_dir
        self.logger = logger
        self.encoding = encoding
        
        
    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            try:
                t2_temp = self.get_random_line()
                t2 = t2_temp
                label = 1
            except:
                label = 0

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        
        sample = self.sample_to_doc[item]
        t1 = self.all_docs[sample["doc_id"]][sample["line"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
        # used later to avoid random nextSentence from same doc
        self.current_doc = sample["doc_id"]
        return t1, t2


    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            rand_doc_idx = random.randint(0, len(self.all_docs)-1)
            rand_doc = self.all_docs[rand_doc_idx]
            line = rand_doc[random.randrange(len(rand_doc))]
            
            # check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def _create_examples(self, lm_file, set_type):
        
        # temporary state variables
        self.sample_to_doc = []
        self.all_docs = []
        self.corpus_lines = 0
        self.current_doc = None
        self.current_random_doc = None
        examples = []
        doc = []
        
        corpus_lines = 0
        with open(lm_file, "r", encoding=self.encoding) as f:
            lines = f.readlines()
            self.logger.info("loading dataset: {}".format(set_type))
            for line in progress_bar(lines):
                line = line.strip()
                if line == "":
                    self.all_docs.append(doc)
                    doc = []
                    # remove last added sample because there won't be a subsequent line anymore in the doc
                    self.sample_to_doc.pop()
                else:
                    # store as one sample
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc)}
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

        # if last row in file is not empty
        if self.all_docs[-1] != doc:
            self.all_docs.append(doc)
            self.sample_to_doc.pop()

        self.num_docs = len(self.all_docs)
        self.items = self.all_docs
        
        for index, sample in enumerate(self.sample_to_doc):
            t1, t2, is_next_label = self.random_sent(index)
            
            examples.append(LMInputExample(guid="{}_{}".format(set_type, str(index)), text_a=t1, text_b=t2, is_next=is_next_label))
        
        return examples
    
    # Get training examples
    def get_train_examples(self, filename='lm_train.txt'):
        lm_file = str(self.data_dir/filename)
        
        return self._create_examples(lm_file, 'train')
    
    # Get validation examples
    def get_dev_examples(self, filename='lm_val.txt'):
        lm_file = str(self.data_dir/filename)
        
        return self._create_examples(lm_file, 'dev')
        

# DataBunch object for language models
class BertLMDataBunch(object):
    
    
    @staticmethod
    def from_raw_corpus(data_dir, text_list,  tokenizer, batch_size_per_gpu=32, max_seq_length=512, multi_gpu=True, test_size=0.1, model_type='bert', logger=None, clear_cache=False, no_cache=False):
        
        train_file = "lm_train.txt"
        val_file = "lm_val.txt"
    
        train_list, val_list = train_test_split(
            text_list, test_size=test_size, shuffle=True)
        # Create train corpus
        create_corpus(train_list, str(data_dir/train_file), logger=logger)

        # Create val corpus
        create_corpus(val_list, str(data_dir/val_file), logger=logger)

        return BertLMDataBunch(data_dir, tokenizer, 
                               train_file=train_file, val_file=val_file,
                               batch_size_per_gpu=batch_size_per_gpu, 
                               max_seq_length=max_seq_length, 
                               multi_gpu=multi_gpu,
                               model_type=model_type,
                               logger=logger,
                               clear_cache=clear_cache, no_cache=no_cache)

    def __init__(self, data_dir, tokenizer, train_file='lm_train.txt', val_file='lm_val.txt',
                 batch_size_per_gpu=32, max_seq_length=512, multi_gpu=True, model_type='bert', logger=None, clear_cache=False, no_cache=False):
        
        
        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        # Instantiate correct tokenizer if the tokenizer name is passed instead of object
        if isinstance(tokenizer, str):
            _, _, tokenizer_class = MODEL_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer, do_lower_case=('uncased' in tokenizer))
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.data_dir = data_dir
        self.cache_dir = data_dir/'lm_cache'
        self.no_cache = no_cache
        self.model_type = model_type
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.n_gpu = 1
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        
        processor = LMTextProcessor(data_dir, self.logger)
        
        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        
        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}'.format(
                self.model_type,
                'train',
                str(self.max_seq_length)))

            if os.path.exists(cached_features_file) == False:
                train_examples = processor.get_train_examples(train_file)

            train_dataset = self.get_dataset_from_examples(train_examples, 'train')

            self.train_batch_size = self.batch_size_per_gpu * \
                max(1, self.n_gpu)
            
            train_sampler = RandomSampler(train_dataset)
            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)
        

        if val_file:
            # Validation DataLoader
            val_examples = None
            cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}'.format(
                self.model_type,
                'dev',
                str(self.max_seq_length)))

            if os.path.exists(cached_features_file) == False:
                val_examples = processor.get_dev_examples(val_file)

            val_dataset = self.get_dataset_from_examples(val_examples, 'dev')

            # no grads necessary, hence double val batch size
            self.val_batch_size = self.batch_size_per_gpu * \
                2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size)
    
    # Get the dataset from the examples
    def get_dataset_from_examples(self, examples, set_type='train', is_test=False, no_cache=False):

        cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}'.format(
            self.model_type,
            set_type,
            str(self.max_seq_length)))

        if os.path.exists(cached_features_file) and no_cache == False:
            self.logger.info(
                "Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            # Create tokenized and numericalized features
            features = convert_examples_to_features(
                examples,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                cls_token_at_end=bool(self.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                logger=self.logger)

            # Create folder if it doesn't exist
            self.cache_dir.mkdir(exist_ok=True)
            if self.no_cache == False or no_cache == False:
                self.logger.info(
                    "Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor(
            [f.lm_label_ids for f in features], dtype=torch.long)
        all_is_next = torch.tensor(
            [f.is_next for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_is_next)
        
        return dataset

           

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label



def convert_examples_to_features(examples, max_seq_length, tokenizer, 
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 logger=None):
    features = []
    """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            if logger:
                logger.info("Writing example {} of {}".format(ex_index, len(examples)))
        
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        t1_random, t1_label = random_word(tokens_a, tokenizer)
        t2_random, t2_label = random_word(tokens_b, tokenizer)
        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
            lm_label_ids = ([-1] * padding_length) +  lm_label_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)
            lm_label_ids = lm_label_ids + ([-1] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length

        features.append(LMInputFeatures(input_ids=input_ids,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids,
                                   lm_label_ids=lm_label_ids,
                                   is_next=example.is_next))
    
    return features

def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    t1_random, t1_label = random_word(tokens_a, tokenizer)
    t2_random, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

#     assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    features = LMInputFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               lm_label_ids=lm_label_ids,
                               is_next=example.is_next)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
