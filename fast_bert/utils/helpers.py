import importlib.util

_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_tpu_available():
    if not _torch_available:
        return False
    # This test is probably enough, but just in case, we unpack a bit.
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if importlib.util.find_spec("torch_xla.core") is None:
        return False
    return importlib.util.find_spec("torch_xla.core.xla_model") is not None
