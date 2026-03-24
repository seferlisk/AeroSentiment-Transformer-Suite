import torch.nn as nn


class TransformerClassifier(nn.Module):
    """
    OOP Wrapper for Transformer-based classification.
    By taking backbone as an argument, it allows for easy swapping between BERT, RoBERTa, and others.
    """

    def __init__(self, backbone, num_labels=3, dropout_rate=0.3):
        super(TransformerClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_rate)
        # The hidden size varies by model; BertModel.config.hidden_size is standard
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # We extract the pooled output (CLS token representation)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Handle cases where model returns pooler_output vs last_hidden_state
        pooled_output = outputs[1] if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]

        output = self.dropout(pooled_output)
        return self.classifier(output)