import os
import io
import json
import pickle
import sys
import signal
import traceback
import re
import flask
import pandas as pd
import torch
from collections import OrderedDict

from fast_bert.prediction import BertClassificationPredictor

from fast_bert.utils.spellcheck import BingSpellCheck
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

prefix = "/opt/ml/"

# PATH = Path(os.path.join(prefix, "model"))
PATH = os.path.join(prefix, "model")

MODEL_PATH = os.path.join(PATH, "pytorch_model.bin")

# request_text = None


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_predictor_model(cls):

        # print(cls.searching_all_files(PATH))
        # Get model predictor
        if cls.model is None:
            with open(os.path.join(PATH, "model_config.json")) as f:
                model_config = json.load(f)

            predictor = BertClassificationPredictor(
                os.path.join(PATH, "model_out"),
                label_path=PATH,
                multi_label=bool(model_config["multi_label"]),
                model_type=model_config["model_type"],
                do_lower_case=bool(model_config["do_lower_case"]),
            )
            cls.model = predictor

        return cls.model

    @classmethod
    def predict(cls, text, bing_key=None):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        predictor_model = cls.get_predictor_model()
        if bing_key:
            spellChecker = BingSpellCheck(bing_key)
            text = spellChecker.spell_check(text)
        prediction = predictor_model.predict(text)

        return prediction

    @classmethod
    def predict_batch(cls, texts):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        predictor_model = cls.get_predictor_model()
        output_labels_count = int(
            os.environ.get(
                "OUTPUT_LABELS_COUNT", len(predictor_model.learner.data.labels)
            )
        )

        print("output_labels_count", output_labels_count)

        predictions = predictor_model.predict_batch(texts)
        return cls.process_batch_results(
            texts, predictions, labels_count=output_labels_count
        )

    @classmethod
    def searching_all_files(cls, directory: Path):
        file_list = []  # A list for storing files existing in directories

        for x in directory.iterdir():
            if x.is_file():
                file_list.append(str(x))
            else:
                file_list.append(cls.searching_all_files(x))

        return file_list

    @classmethod
    def process_batch_results(cls, texts, results, labels_count=None):
        processed_results = []
        for i, result in enumerate(results):
            processed = OrderedDict()
            processed["text"] = texts[i]
            result = result[:labels_count] if labels_count else result
            for index, label in enumerate(result):
                processed["label_{}".format(index + 1)] = label[0]
                processed["confidence_{}".format(index + 1)] = label[1]
            processed_results.append(processed)

        return processed_results


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_predictor_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/execution-parameters", methods=["GET"])
def get_execution_parameters():
    params = {
        "MaxConcurrentTransforms": 3,
        "BatchStrategy": "MULTI_RECORD",
        "MaxPayloadInMB": 6,
    }
    return flask.Response(
        response=json.dumps(params), status="200", mimetype="application/json"
    )


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    text = None

    if flask.request.content_type == "application/json":
        print("calling json launched")
        data = flask.request.get_json(silent=True)

        text = data["text"]
        try:
            bing_key = data["bing_key"]
        except Exception:
            bing_key = None

        # Do the prediction
        predictions = ScoringService.predict(text, bing_key)
        result = json.dumps(predictions[:10])
        return flask.Response(response=result, status=200, mimetype="application/json")

    elif flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        df = pd.read_csv(io.StringIO(data), header="infer")
        predictions = ScoringService.predict_batch(list(df["text"].values))

        out = io.StringIO()
        pd.DataFrame(predictions).to_csv(out, index=False)
        result = out.getvalue()
        return flask.Response(response=result, status=200, mimetype="text/csv")

    elif flask.request.content_type == "text/plain":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        # convert txt file into list of texts
        texts = []
        for line in s:
            texts.append(line)
        predictions = ScoringService.predict_batch(texts)
        out = io.StringIO()
        pd.DataFrame(predictions).to_csv(out, index=False)
        result = out.getvalue()
        return flask.Response(response=result, status=200, mimetype="text/csv")

    else:
        return flask.Response(
            response="This predictor only supports JSON, txt or CSV data",
            status=415,
            mimetype="text/plain",
        )
