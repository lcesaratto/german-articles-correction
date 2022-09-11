from datasets import load_metric
import torch
import warnings
warnings.filterwarnings("ignore")

from german_grammar_checker.model_preparation import Model
from german_grammar_checker.helper_functions import get_device
from german_grammar_checker.data_preparation import DataPreparator


TEST_DATA_PATH = "data/eval.csv"
MODEL_NAME = "bert-base-german-cased"
PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict.pt"
BATCH_SIZE = 16


class BertForGrammarCorrectionEvaluator:
    def __init__(self, model_name, batch_size, pretrained_model_path):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = get_device()

        self.model = Model()
        self.model = self.model.to(self.device)

    def evaluate_on_test_data(self, test_data_path):

        test_dataloader = DataPreparator.get_dataloaders(test_data_path, self.model_name, self.batch_size)

        testing_stats = []
        try:
            total_test_loss = 0
            metric = load_metric("f1")

            self.model.eval()

            for batch in test_dataloader:

                parameters = {
                    "input_ids" : batch[0].to(self.device),
                    "attention_mask" :  batch[1].to(self.device), 
                    "labels" : batch[2].to(self.device)
                }
                with torch.no_grad():
                    outputs = self.model(**parameters)
                
                logits = outputs.logits
                loss = outputs.loss
                total_test_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=parameters["labels"])

            testing_stats.append({
                "test_loss": total_test_loss/len(test_dataloader),
                "test_f1_score": metric.compute()
            })

            print(f"\nAvg test loss:  {testing_stats[0]['test_loss']}")
            print(f"F1 test score:  {testing_stats[0]['test_f1_score']}\n")

        except RuntimeError as e:
            print(e)

        print("\nTesting results: ", testing_stats)

bert_for_checking_grammar = BertForGrammarCorrectionEvaluator(MODEL_NAME, BATCH_SIZE, PRETRAINED_MODEL_PATH)
bert_for_checking_grammar.evaluate_on_test_data(TEST_DATA_PATH)
