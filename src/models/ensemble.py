import torch

class SentimentEnsemble:
    def __init__(self, model_db, model_mini, classes, weights=None):
        """
        Initializes the ensemble with two models and their respective weights.

        Args:
            model_db: The trained DistilBERT model.
            model_mini: The trained MiniLM model.
            classes (list): List of class names (e.g., ['negative', 'neutral', 'positive']).
            weights (list): Weighting for [DistilBERT, MiniLM].
        """
        if weights is None:
            weights = [0.6, 0.4]
        self.model_db = model_db
        self.model_mini = model_mini
        self.classes = classes
        self.weights = weights

    def predict(self, text, tokenizer_db, tokenizer_mini, device):
        # 1. Get Logits from both models
        logits_db = self._get_logits(text, self.model_db, tokenizer_db, device)
        logits_mini = self._get_logits(text, self.model_mini, tokenizer_mini, device)

        # 2. Soft Voting: Weighted average of Softmax probabilities
        # Soft voting outperforms hard voting by incorporating prediction confidence.
        probs_db = torch.softmax(logits_db, dim=1)
        probs_mini = torch.softmax(logits_mini, dim=1)

        ensemble_probs = (self.weights[0] * probs_db) + (self.weights[1] * probs_mini)

        # 3. Final Prediction
        _, prediction = torch.max(ensemble_probs, dim=1)
        return self.classes[prediction.item()]  # FIX: Access via self.classes

    def _get_logits(self, text, model, tokenizer, device):
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            return model(inputs['input_ids'], inputs['attention_mask'])