from transformers import AutoTokenizer, AutoModel

class ModelFactory:
    """
    Static factory to generate models and tokenizers based on string names.
    Factory Pattern implementation to dynamically fetch Hugging Face components.
    """
    @staticmethod
    def get_model_and_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)
        return backbone, tokenizer