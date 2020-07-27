import pandas as pd
import json
import logging
import os
import torch
from torch import nn
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from pathlib import Path
import pickle
from filelock import FileLock
import re
import shutil
from sklearn.model_selection import train_test_split

from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from transformers import PreTrainedTokenizer, AutoTokenizer


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NerDataset(Dataset):
    """
        This will be superseded by a framework-agnostic approach
        soon.
        """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_dir: str,
        file_name: str,
        tokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
        logger=logging.getLogger(__name__),
    ):
        # # Load data features from cache or dataset file
        # cached_features_file = os.path.join(
        #     data_dir,
        #     "cached_{}_{}_{}".format(
        #         mode.value, tokenizer.__class__.__name__, str(max_seq_length)
        #     ),
        # )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        # lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):

        # if os.path.exists(cached_features_file) and not overwrite_cache:
        #     logger.info(f"Loading features from cached file {cached_features_file}")
        #     self.features = torch.load(cached_features_file)
        # else:
        logger.info(f"Creating features from dataset file at {data_dir}")
        examples = read_examples_from_file(data_dir, file_name, mode)
        self.features = convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
            logger=logger,
        )
        # logger.info(f"Saving features into cached file {cached_features_file}")
        # torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(
    data_dir, file_name, mode: Union[Split, str]
) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, file_name)
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid=f"{mode}-{guid_index}", words=words, labels=labels
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels)
            )
    return examples


class BertNERDataBunch(object):
    @classmethod
    def from_jsonl(
        cls,
        data_dir,
        file_name,
        tokenizer,
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        backend="nccl",
        model_type="bert",
        logger=logging.getLogger(),
        clear_cache=False,
        no_cache=False,
        use_fast_tokenizer=True,
        custom_sampler=None,
        train_size=0.8,
        random_state=None,
    ):
        DATA_PATH = Path(data_dir)

        labels = []

        with open(DATA_PATH / file_name, "r") as f:
            lines = f.readlines()
            label_lines = [json.loads(line) for line in lines]
            for line in label_lines:
                for label in line["labels"]:
                    labels.append(label[2])

        train, val = train_test_split(
            lines, train_size=train_size, random_state=random_state
        )

        modified_train = convert_data(train)
        modified_val = convert_data(val)

        modified_train = flatten_all(modified_train)
        json_to_text(modified_train, str(DATA_PATH / "train.txt"))

        modified_val = flatten_all(modified_val)
        json_to_text(modified_val, str(DATA_PATH / "val.txt"))

        with open(DATA_PATH / "labels.txt", "w") as f:
            for label in set(labels):
                f.write("B-{}\n".format(label))
                f.write("I-{}\n".format(label))

        return BertNERDataBunch(
            data_dir,
            tokenizer=tokenizer,
            batch_size_per_gpu=batch_size_per_gpu,
            max_seq_length=max_seq_length,
            multi_gpu=multi_gpu,
            backend=backend,
            model_type=model_type,
            logger=logger,
            clear_cache=clear_cache,
            no_cache=no_cache,
            use_fast_tokenizer=use_fast_tokenizer,
            custom_sampler=custom_sampler,
        )

    def __init__(
        self,
        data_dir,
        tokenizer,
        train_file="train.txt",
        val_file="val.txt",
        label_file="labels.txt",
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        backend="nccl",
        model_type="bert",
        logger=logging.getLogger(),
        clear_cache=False,
        no_cache=False,
        processor_name="ner",
        use_fast_tokenizer=True,
        custom_sampler=None,
    ):
        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        # instantiate auto tokenizer if not already instantiated
        if isinstance(tokenizer, str):
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, use_fast=use_fast_tokenizer
            )

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.cache_dir = data_dir / "cache"
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.no_cache = no_cache
        self.model_type = model_type
        self.logger = logger
        self.custom_sampler = custom_sampler
        self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        self.labels = get_labels("{}/{}".format(str(data_dir), label_file))
        self.label_map = {i: label for i, label in enumerate(self.labels)}

        if train_file:
            train_dataset = NerDataset(
                data_dir=data_dir,
                file_name=train_file,
                tokenizer=tokenizer,
                labels=self.labels,
                model_type=self.model_type,
                max_seq_length=max_seq_length,
                overwrite_cache=clear_cache,
                mode=Split.train,
            )

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)

            if self.custom_sampler is not None:
                train_sampler = self.custom_sampler
            else:
                train_sampler = RandomSampler(train_dataset)

            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
            )

        if val_file:
            val_dataset = NerDataset(
                data_dir=data_dir,
                file_name=val_file,
                tokenizer=tokenizer,
                labels=self.labels,
                model_type=self.model_type,
                max_seq_length=max_seq_length,
                overwrite_cache=clear_cache,
                mode=Split.dev,
            )

            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size
            )


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    logger=logging.getLogger(__name__),
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend(
                    [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
                )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return [
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]


def convert_data(lines, label_name="labels"):
    modified_lines = []
    for line in lines:
        line = json.loads(line)
        if "labels" in line:
            line[label_name] = line.pop("labels")
        else:
            line[label_name] = []
            continue

        tmp_ents = []
        for e in line[label_name]:
            tmp_ents.append((e[0], e[1], e[2]))
            # tmp_ents.append({"start": e[0], "end": e[1], "label": e[2]})

            line[label_name] = tmp_ents

        if len(line["text"]) > 3:
            modified_lines.append(
                json.dumps({label_name: line[label_name], "text": line["text"]})
            )

    return modified_lines


def flatten(data):
    data = json.loads(data)
    output_text = []
    beg_index = 0
    end_index = 0

    text = data["text"]
    all_labels = sorted(data["labels"])

    for ind in range(len(all_labels)):
        next_label = all_labels[ind]
        output_text += [
            (label_word, "O")
            for label_word in text[end_index : next_label[0]].strip().split()
        ]

        label = next_label
        beg_index = label[0]
        end_index = label[1]
        label_text = text[beg_index:end_index]
        output_text += [
            (label_word, "B-" + label[2]) if not i else (label_word, "I-" + label[2])
            for i, label_word in enumerate(label_text.split(" "))
        ]

    output_text += [
        (label_word, "O") for label_word in text[end_index:].strip().split()
    ]
    return output_text


def flatten_all(datas):
    return [flatten(data) for data in datas]


def json_to_text(jsons, output_filename):
    with open(output_filename, "w") as f:
        for each_json in jsons:
            for line in each_json:
                f.writelines(" ".join(line) + "\n")
            f.writelines("\n")
