# flake8: noqa: E501

import importlib.util


def is_torch_available():
    """
    Check if PyTorch is available.
    """
    return importlib.util.find_spec("torch") is not None


def is_torch_cuda_available():
    """
    Check if CUDA is available.
    """
    if is_torch_available():
        import torch  # pylint: disable=import-outside-toplevel

        return torch.cuda.is_available()
    return False


def is_detectron2_available():
    """
    Check if Detectron2 is available.
    """
    return importlib.util.find_spec("detectron2") is not None


def is_flash_attention_available():
    """
    Check if Flash Attention is available.
    """
    return importlib.util.find_spec("flash_attn") is not None


def is_qwen2_available():
    """
    Check if the transformers library and Qwen2 are available.
    """
    try:
        importlib.import_module("transformers")
        from transformers import (  # pylint: disable=import-outside-toplevel,unused-import
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        return True
    except ImportError:
        return False


def is_paddle_available():
    """
    Check if PaddlePaddle is available.
    """
    return importlib.util.find_spec("paddlepaddle") is not None


def is_paddleocr_available():
    """
    Check if PaddleOCR is available.
    """
    return importlib.util.find_spec("paddleocr") is not None


def is_vllm_available():
    """
    Check if VLLM is available.
    """
    return importlib.util.find_spec("vllm") is not None
