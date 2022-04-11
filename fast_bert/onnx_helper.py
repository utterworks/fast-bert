from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger()


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

    assert (
        provider in get_all_providers()
    ), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


def load_model(model_path: Path):
    try:
        quantized_model = create_model_for_provider(
            model_path.as_posix(), "CPUExecutionProvider"
        )
        return quantized_model
    except Exception as e:
        logger.error(e)
        raise e
