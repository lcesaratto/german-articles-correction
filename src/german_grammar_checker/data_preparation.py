import pandas as pd
from transformers import BertTokenizer
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


class DataPreparator:
    def __init__(self, model_name, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def _load_dataset(self, data_path) -> pd.DataFrame:
        return pd.read_csv(data_path, nrows=None)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["sentence", "wrong_sentence"]]
        return df

    def _create_tensors(self, df: pd.DataFrame):
        wrong_sentences = df["wrong_sentence"].values
        sentences = df["sentence"].values

        input_ids = []
        attention_masks = []
        print("Tokenizing training data...")
        for sent in tqdm(wrong_sentences):
            encoded_dict = self.tokenizer(
                sent,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        labels = []
        print("Tokenizing labels...")
        for sent in tqdm(sentences):
            encoded_dict = self.tokenizer(
                sent,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_attention_mask=False,
                return_tensors='pt',
            )
            labels.append(encoded_dict['input_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.cat(labels, dim=0)

        return input_ids, attention_masks, labels

    def _create_dataloader(self, input_ids, attention_masks, labels):
        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(dataset, shuffle=False,
                                batch_size=self.batch_size)

        return dataloader

    def get_dataloaders(self, data_path):
        df = self._load_dataset(data_path)
        df = self._prepare_data(df)
        input_ids, attention_masks, labels = self._create_tensors(df)
        dataloader = self._create_dataloader(
            input_ids, attention_masks, labels)

        return dataloader
