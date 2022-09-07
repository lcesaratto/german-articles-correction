import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

from german_grammar_checker.preprocessor import Preprocessor


def _load_dataset(data_path) -> pd.DataFrame:
    return pd.read_csv(data_path, sep=',')

def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True, errors="ignore")
    df.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)

    pipeline = ['hyperlinks', 'mentions', 'hashtags', 'retweet', 'repetitions', 'emojis', 'smileys', 'spaces']
    preprocessor = Preprocessor(pipeline)
    df["text"] = df["text"].apply(preprocessor)

    df["label"] = df["label"].apply(lambda x: 0 if x == "Neutral" else 1)

    return df

def _create_tensors(df: pd.DataFrame, model_name: str):
    sentences = df["text"].values
    labels = df["label"].values.astype(int)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer(
                            sent,
                            truncation=True,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]', True by default
                            max_length = 128,           # Pad & truncate all sentences.
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_masks, labels

def _create_dataloader(input_ids, attention_masks, labels, batch_size):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloader

def get_dataloaders(data_path,  model_name, batch_size):
    df = _load_dataset(data_path)
    df = _prepare_data(df)
    input_ids, attention_masks, labels = _create_tensors(df, model_name)
    dataloader = _create_dataloader(input_ids, attention_masks, labels, batch_size)

    return dataloader