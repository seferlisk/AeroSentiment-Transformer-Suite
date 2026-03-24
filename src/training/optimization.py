class FineTuningManager:
    """
    Handles freezing/unfreezing logic for Transformer layers.
    """
    @staticmethod
    def freeze_backbone(model):
        """Freezes all layers in the transformer backbone."""
        for param in model.backbone.parameters():
            param.requires_grad = False

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

        # Freeze all first
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always keep the classifier head unfrozen
        for param in model.classifier.parameters():
            param.requires_grad = True