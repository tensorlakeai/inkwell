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