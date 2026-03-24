import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data.dataset import TweetDataset


class DataManager:
    """
    Orchestrates data splitting and DataLoader creation.
    It includes the label mapping derived from the dataset (negative, neutral, positive).
    """
    LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __init__(self, csv_path, tokenizer, batch_size=16, max_len=128, test_size=0.2, val_size=0.1):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.test_size = test_size
        self.val_size = val_size

        # Preprocess labels
        self.df['sentiment_code'] = self.df['airline_sentiment'].map(self.LABEL_MAP)

    def prepare_loaders(self):
        # Initial split: Train+Val vs Test
        df_train_val, df_test = train_test_split(
            self.df, test_size=self.test_size, random_state=42, stratify=self.df['sentiment_code']
        )

        # Second split: Train vs Validation
        df_train, df_val = train_test_split(
            df_train_val, test_size=self.val_size, random_state=42, stratify=df_train_val['sentiment_code']
        )

        train_loader = self._create_loader(df_train)
        val_loader = self._create_loader(df_val)
        test_loader = self._create_loader(df_test)

        return train_loader, val_loader, test_loader

    def _create_loader(self, data_frame):
        dataset = TweetDataset(
            texts=data_frame.text.to_numpy(),
            labels=data_frame.sentiment_code.to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)