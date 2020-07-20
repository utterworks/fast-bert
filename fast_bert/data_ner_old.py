import pandas as pd
import os
import torch
from pathlib import Path
import pickle
import logging
import re

import shutil

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer)
}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        
def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerCustomProcessor(DataProcessor):

    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None
    
    def extract_entities(self, text):
        regex = r"\[([^\[\]]*)\]\((\w*)\)"

        matches = re.finditer(regex, text, re.MULTILINE)
        temp_text = text
        for _, match in enumerate(matches):
            groups = match.groups()
            entity_text = groups[0]
            entity_name = groups[1]

            entity_parts = entity_text.split(' ')
            entity_string_components = []
            for i, entity_part in enumerate(entity_parts):
                if i == 0:
                    prefix = 'B'
                else:
                    prefix = 'I'
                entity_formatted_name = "{}-{}".format(prefix, entity_name)
                entity_string_components.append(entity_formatted_name)

            text = text.replace(match.group(), entity_text)
            temp_text = temp_text.replace(match.group(), "#{}".format('#'.join(entity_string_components)))


        # create labels
        labels = []
        words = temp_text.split(' ')
        for word in words:
            if word[0] == '#':
                subwords = word[1:].split("#")
                labels.extend(subwords)
            else:
                labels.append('O')

        return labels, text

    def get_train_examples(self, filename='train.csv', text_col='text', size=-1):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df, "train", text_col=text_col)
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df.sample(size), "train", text_col=text_col)

    def get_dev_examples(self, filename='val.csv', text_col='text',size=-1):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df, "dev", text_col=text_col)
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df.sample(size), "dev", text_col=text_col)

    def get_test_examples(self, filename='val.csv', text_col='text', size=-1):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        if size == -1:
            return self._create_examples(data_df, "test",  text_col=text_col)
        else:
            return self._create_examples(data_df.sample(size), "test", text_col=text_col)

    def get_labels(self, filename='labels.csv'):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(
                self.label_dir, filename), header=None)[0].astype('str').values)
            self.labels.extend(['[CLS]', '[SEP]'])
            self.labels.insert(0, 'O')
        return self.labels

    def _create_examples(self, df, set_type, text_col):
        """Creates examples for the training and dev sets."""
        if set_type == "test":
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=row[text_col], label=None), axis=1))
        else:
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=self.extract_entities(row[text_col])[1],
                                                          label=self.extract_entities(row[text_col])[0]), axis=1))

    
    
class NerColProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(self, filename='val.csv', text_col='text', label_col='label', size=-1):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.txt")), "train")
    
    def get_dev_examples(self, filename='val.csv', text_col='text', label_col='label', size=-1):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "valid.txt")), "dev")
    
    def get_test_examples(self, filename='val.csv', text_col='text', label_col='label', size=-1):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.txt")), "test")
    
    def get_labels(self, filename='labels.csv'):
        
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(self.label_dir, filename), header=None)[0].astype('str').values)
        
        return self.labels

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
        

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode='classification',
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, logger=None):
    
    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            
            if example.label:
                label_1 = labellist[i]
                
            for m in range(len(token)):
                if m == 0:
                    if example.label:
                        labels.append(label_1)
                        label_mask.append(1)
                    valid.append(1)
                else:
                    valid.append(0)
                    
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            
            if example.label:
                labels = labels[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
        
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append(cls_token)
        segment_ids.append(0)
        valid.insert(0,1)
        
        if example.label:
            label_mask.insert(0,1)
            label_ids.append(label_map[cls_token])
        
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append(sep_token)
        segment_ids.append(0)
        valid.append(1)
        
        if example.label:
            label_mask.append(1)
            label_ids.append(label_map[sep_token])
            label_mask = [1] * len(label_ids)
        
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            
            if example.label:
                label_ids.append(0)
                label_mask.append(0)
            valid.append(1)
            
        if example.label:
            while len(label_ids) < max_seq_length: 
                label_ids.append(0)
                label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        assert len(valid) == max_seq_length
        
        if example.label:
            assert len(label_mask) == max_seq_length
            assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))
        
        if example.label == None:
            label_mask = None
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


class BertNERDataBunch(object):

    def __init__(self, data_dir, label_dir, tokenizer, train_file='train.csv', val_file='val.csv', test_data=None,
                 label_file='labels.csv', text_col='text', batch_size_per_gpu=16, max_seq_length=512,
                 multi_gpu=True, multi_label=False, backend="nccl", model_type='bert', logger=None, clear_cache=False, no_cache=False, processor_name='ner'):
        
        if isinstance(tokenizer, str):
            _,_,tokenizer_class = MODEL_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = tokenizer_class.from_pretrained(tokenizer, do_lower_case=('uncased' in tokenizer))

        self.tokenizer = tokenizer  
        self.data_dir = data_dir
        self.cache_dir = data_dir/'cache'    
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.multi_label = multi_label
        self.n_gpu = 0
        self.no_cache = no_cache
        self.model_type = model_type
        self.output_mode = 'classification'
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        
        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        
        if processor_name == 'col':
            processor = NerColProcessor(data_dir, label_dir)
        else:
            processor = NerCustomProcessor(data_dir, label_dir)

        self.labels = processor.get_labels(label_file)
        self.label_map = {i : label for i, label in enumerate(self.labels)}
        
        if train_file:
            # Train DataLoader
            train_examples = processor.get_train_examples(
                train_file, text_col=text_col)  

            train_dataset, _ = self.get_dataset_from_examples(train_examples, 'train')

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            train_sampler = RandomSampler(train_dataset)
            self.train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)
            

        if val_file:
            # Validation DataLoader
            val_examples = processor.get_dev_examples(
                val_file, text_col=text_col)
            
            val_dataset, _ = self.get_dataset_from_examples(val_examples, 'dev')
            
            self.val_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset) 
            self.val_dl = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.val_batch_size)
            
        
        if test_data:
            # Test set loader for predictions 
            test_examples = []
            input_data = []

            for index, text in enumerate(test_data):
                test_examples.append(InputExample(index, text))
                input_data.append({
                    'id': index,
                    'text': text
                })


            test_dataset, _ = self.get_dataset_from_examples(test_examples, 'test', is_test=True)
            
            self.test_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            test_sampler = SequentialSampler(test_dataset)
            self.test_dl = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.test_batch_size)

    
    def get_dl_from_texts(self, texts):

        test_examples = []
        input_data = []
        
        for index, text in enumerate(texts):
            test_examples.append(InputExample(index, text, label=None))
            input_data.append({
                'id': index,
                'text': text
            })
        
        test_dataset, features = self.get_dataset_from_examples(test_examples, 'test', is_test=True, ignore_cache=True)
        
        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(test_dataset, sampler=test_sampler, batch_size=self.batch_size_per_gpu), features

    
    def get_dataset_from_examples(self, examples, set_type='train', is_test=False, ignore_cache=False):
        
        
        cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}'.format(
            set_type,
            'multi_label' if self.multi_label else 'multi_class',
            str(self.max_seq_length)))
        
        if os.path.exists(cached_features_file) and ignore_cache==False:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            # Create tokenized and numericalized features 
            features = convert_examples_to_features(
                    examples, 
                    label_list=self.labels, 
                    max_seq_length=self.max_seq_length, 
                    tokenizer=self.tokenizer, 
                    output_mode=self.output_mode,
                    cls_token_at_end=bool(self.model_type in ['xlnet']), # xlnet has a cls token at the end
                    cls_token=self.tokenizer.cls_token,
                    sep_token=self.tokenizer.sep_token,
                    cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                    pad_on_left=bool(self.model_type in ['xlnet']),                 # pad on the left for xlnet
                    pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                    logger=self.logger)
            
            self.cache_dir.mkdir(exist_ok=True)  # Creaet folder if it doesn't exist
            if self.no_cache == False:
                self.logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        
        if is_test == False: # labels not available for test set
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)    
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        else:
            all_label_ids = []
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        
        
        return dataset, features