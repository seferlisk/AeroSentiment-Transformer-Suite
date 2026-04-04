import torch

def get_predictions(model, data_loader, device):
    """
    Helper to collect all predictions and true labels from a loader.
    """
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())

    return predictions, real_values

def predict_sentiment(text, model, tokenizer, device, classes, max_len=64):
    """
     Predicts sentiment for a single string.

     Args:
        text (str): The input text.
        model: The trained Transformer model.
        tokenizer: The corresponding tokenizer.
        device: torch.device (CPU or CUDA).
        classes (list): The list of labels.
        max_len (int): Maximum sequence length.
    """
    model.eval()

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return classes[preds.item()]