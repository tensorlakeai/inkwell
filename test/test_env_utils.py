# Tests to make sure that the library works without some dependencies

# pylint: disable=import-outside-toplevel

import logging
import unittest
from unittest.mock import patch

_logger = logging.getLogger(__name__)


class TestEnvUtils(unittest.TestCase):

    def setUp(self):
        _logger.info("Running test: %s", self._testMethodName)

    @patch("inkwell.utils.env_utils.is_vllm_available", return_value=False)
    def test_is_vllm_available(self, mock_is_vllm_available):
        mock_is_vllm_available.return_value = None
        import inkwell

        inkwell.utils.env_utils.is_vllm_available()

    @patch(
        "inkwell.utils.env_utils.is_torch_cuda_available", return_value=False
    )
    def test_is_torch_cuda_available(self, mock_is_torch_cuda_available):
        mock_is_torch_cuda_available.return_value = None
        import inkwell

        inkwell.utils.env_utils.is_torch_cuda_available()

    @patch(
        "inkwell.utils.env_utils.is_flash_attention_available",
        return_value=False,
    )
    def test_is_flash_attention_available(
        self, mock_is_flash_attention_available
    ):
        mock_is_flash_attention_available.return_value = None
        import inkwell

        inkwell.utils.env_utils.is_flash_attention_available()
