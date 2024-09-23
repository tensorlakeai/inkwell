from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS,
    RobertaConverter,
)

from .configuration_layoutlmv3 import LayoutLMv3ConfigLocal
from .modeling_layoutlmv3 import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Model,
)
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast

AutoConfig.register("layoutlmv3local", LayoutLMv3ConfigLocal)
AutoModel.register(LayoutLMv3ConfigLocal, LayoutLMv3Model)
AutoModelForTokenClassification.register(
    LayoutLMv3ConfigLocal, LayoutLMv3ForTokenClassification
)
AutoModelForQuestionAnswering.register(
    LayoutLMv3ConfigLocal, LayoutLMv3ForQuestionAnswering
)
AutoModelForSequenceClassification.register(
    LayoutLMv3ConfigLocal, LayoutLMv3ForSequenceClassification
)
AutoTokenizer.register(
    LayoutLMv3ConfigLocal,
    slow_tokenizer_class=LayoutLMv3Tokenizer,
    fast_tokenizer_class=LayoutLMv3TokenizerFast,
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
