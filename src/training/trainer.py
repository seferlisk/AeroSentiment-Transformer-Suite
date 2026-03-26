import os
import torch
import torch.nn as nn
from tqdm import tqdm

class SentimentTrainer:
    """
    This class encapsulates the entire training logic. It is designed to be decoupled from the specific model architecture.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss().to(device)

        # History for plotting
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self):
        self.model.train()
        losses = []
        # Initialize as a float to avoid type issues and track as a standard scalar
        correct_predictions = 0

        for data in tqdm(self.train_loader, desc="Training"):
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            labels = data['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, labels)

            # Use .item() to get a standard Python number
            correct_predictions += torch.sum(preds == labels).item()
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Standard division (Python 3 handles float division automatically)
        avg_acc = correct_predictions / len(self.train_loader.dataset)
        avg_loss = sum(losses) / len(self.train_loader)
        return avg_acc, avg_loss

    def evaluate(self, loader):
        """Accepts any loader (val or test) for modularity"""
        self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in loader:
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, labels)

                # Use .item() to get a standard Python number
                correct_predictions += torch.sum(preds == labels).item()
                losses.append(loss.item())

        avg_acc = correct_predictions / len(loader.dataset)
        avg_loss = sum(losses) / len(loader)
        return avg_acc, avg_loss

    def save_model(self, filename: str):
        """
        Saves the model weights to the 'outputs' folder in the project root.

        Args:
            filename (str): Name of the file (e.g., 'distilbert_airline.pt')
        """
        # Find the root directory relative to this file (src/training/trainer.py)
        # One level up is src/, two levels up is the project root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_file_dir, "../../"))
        save_path = os.path.join(root_dir, "outputs")

        # Create the directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")

        full_path = os.path.join(save_path, filename)

        # Save only the state dictionary
        torch.save(self.model.state_dict(), full_path)
        print(f"Model successfully saved to: {full_path}")