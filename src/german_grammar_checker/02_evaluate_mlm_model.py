from transformers import BertTokenizer, BertForMaskedLM
from datasets import load_metric
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


TEST_DATA_PATH = "data/raw_data_short.csv"
MODEL_NAME = "bert-base-german-cased"
BATCH_SIZE = 16


class BertForMaskedLanguageModeling:
    def __init__(self, model_name, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = self._get_device()
        self.model = self.model.to(self.device)

    def _get_device(self, show_info=True):
        if torch.cuda.is_available():
            device = torch.device("cuda")

            if show_info:
                print('There are %d GPU(s) available.' %
                      torch.cuda.device_count())
                print('We will use the GPU:', torch.cuda.get_device_name(0))

        else:
            device = torch.device("cpu")

            if show_info:
                print('No GPU available, using the CPU instead.')

        return device

    def _load_dataset(self, data_path) -> pd.DataFrame:
        return pd.read_csv(data_path, nrows=10)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["masked_sentence", "masked_token"]]

        return df

    def _create_tensors(self, df: pd.DataFrame):
        sentences = df["masked_sentence"].values
        tokens = df["masked_token"].values

        input_ids = []
        attention_masks = []
        for sent in sentences:
            encoded_dict = self.tokenizer(
                sent,
                max_length=128,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        labels = []
        for token in tokens:
            encoded_dict = self.tokenizer(
                token,
                return_attention_mask=False,
                return_tensors='pt',
            )
            labels.append(encoded_dict['input_ids'][0][1])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long).view(-1, 1)

        return input_ids, attention_masks, labels

    def _create_dataloader(self, input_ids, attention_masks, labels):
        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(dataset, shuffle=False,
                                batch_size=self.batch_size)

        return dataloader

    def _get_dataloader(self, data_path):
        df = self._load_dataset(data_path)
        df = self._prepare_data(df)
        input_ids, attention_masks, labels = self._create_tensors(df)
        dataloader = self._create_dataloader(
            input_ids, attention_masks, labels)

        return dataloader

    def predict(self, test_data_path):
        test_dataloader = self._get_dataloader(test_data_path)

        self.model.eval()

        for idx, batch in enumerate(test_dataloader):

            parameters = {
                "input_ids": batch[0].to(self.device),
                "attention_mask":  batch[1].to(self.device)
            }
            with torch.no_grad():
                outputs = self.model(**parameters)
            logits = outputs.logits

            for i in range(logits.shape[0]):
                mask_token_index = (parameters["input_ids"] == self.tokenizer.mask_token_id)[
                    i].nonzero(as_tuple=True)[0]
                predicted_token_id = logits[i,
                                            mask_token_index].argmax(axis=-1)
                print(pd.read_csv(test_data_path, skiprows=range(
                    1, idx+i+1), nrows=1)["masked_sentence"][0])
                print("PREDICTED: ", self.tokenizer.decode(predicted_token_id))
                print("EXPECTED: ", self.tokenizer.decode(
                    batch[2][i].to(self.device)))


bert_for_masked_lm = BertForMaskedLanguageModeling(MODEL_NAME, BATCH_SIZE)
bert_for_masked_lm.predict(TEST_DATA_PATH)
