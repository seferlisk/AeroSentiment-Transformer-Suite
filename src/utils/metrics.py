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