import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import urllib3
from transformers import AutoModel, AutoTokenizer


def download_pretrained_files(model_name, location):
    try:
        model_path = model_name.replace("/", ":")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(location / model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(location / model_path)
    except Exception as e:
        print(e)
        print("error downloading model {}".format(model_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location_dir",
        default=None,
        type=str,
        required=True,
        help="The location where pretrained model needs to be stored",
    )

    parser.add_argument(
        "--models",
        default=None,
        type=str,
        required=True,
        nargs="*",
        help="download the pretrained models",
    )

    args = parser.parse_args()
    print(args)
    Path(args.location_dir).mkdir(exist_ok=True)

    #    [download_pretrained_files(k, location=Path(args.location_dir))
    #     for k, v in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.items()]
    [
        download_pretrained_files(item, location=Path(args.location_dir))
        for item in args.models
    ]


if __name__ == "__main__":
    main()
