import torch.nn as nn

class FineTuningManager:
    """
    Handles freezing/unfreezing logic for Transformer layers.
    """
    @staticmethod
    def freeze_entire_backbone(model):
        """Freezes all layers in the transformer backbone."""
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Strategy: Entire backbone is FROZEN.")

    @staticmethod
    def unfreeze_last_n_layers(model, n=1):
        """
        Unfreezes the last N layers of the transformer encoder.
        Works for BERT, RoBERTa, and DistilBERT.
        """
        # DistilBERT uses 'transformer.layer', BERT/RoBERTa use 'encoder.layer'
        if hasattr(model.backbone, 'encoder'):
            layers = model.backbone.encoder.layer
        else:
            layers = model.backbone.transformer.layer

        # Freeze everything first to ensure a clean state
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always ensure the classifier head is unfrozen
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(f"Strategy: Unfrozen the last {n} layers + Classifier Head.")

    @staticmethod
    def unfreeze_all(model):
        """Unfreezes every parameter in the model."""
        for param in model.parameters():
            param.requires_grad = True
        print("Strategy: FULL model is unfrozen.")