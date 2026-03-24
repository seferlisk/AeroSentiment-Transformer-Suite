from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
from typing import Tuple

class ModelFactory:
    """
    Static factory to generate models and tokenizers based on string names.
    Factory Pattern implementation to dynamically fetch Hugging Face components.
    """
    @staticmethod
    def get_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        # We add type hints to the variables to satisfy the IDE's static analysis
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        return backbone, tokenizer