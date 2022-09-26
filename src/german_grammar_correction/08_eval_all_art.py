from german_grammar_correction.grammar_correction.data_preparation_all_articles import DataPreparator
from german_grammar_correction.grammar_correction.helper_functions import get_device
from german_grammar_correction.grammar_correction.model_preparation_all_articles import Model
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


class BertForGrammarCorrectionEvaluator:
    def __init__(self, model_name, pretrained_model_path):
        self.model_name = model_name
        self.model_class = Model(model_name)
        self.model_class.init_model()
        self.model_class.load_model_state_dict(pretrained_model_path)
        self.model_class.init_criterion()

        self.device = get_device()
        self.model_class.model.to(self.device)

    def evaluate_on_test_data(self, batch_size, test_data_path):
        data_preparator = DataPreparator(self.model_name, batch_size)
        test_dataloader = data_preparator.get_dataloaders(test_data_path)

        testing_stats = {"step": [], "test_loss": []}

        self.model_class.model.eval()

        for step, batch in enumerate(test_dataloader):
            parameters = {
                "input_ids": batch[0].to(self.device),
                "attention_mask":  batch[1].to(self.device)
            }
            with torch.no_grad():
                output = self.model_class.model(**parameters)

            loss = self.model_class.criterion(
                output.transpose(1, 2),
                batch[2].to(self.device))

            if step % 100 == 0 and step != 0:
                print(
                    f"BATCH {step}/{len(test_dataloader)}:\tTest loss({loss.item()})")

            testing_stats["step"].append(step+1)
            testing_stats["training_loss"].append(loss.item())

        return testing_stats

    def evaluate_on_sentences(self, wrong_sentence):
        data_preparator = DataPreparator(self.model_name, 1)
        sent_dataloader = data_preparator.get_sentence_dataloader(
            wrong_sentence)

        self.model_class.model.eval()

        batch = next(iter(sent_dataloader))
        parameters = {
            "input_ids": batch[0].to(self.device),
            "attention_mask":  batch[1].to(self.device)
        }
        with torch.no_grad():
            output = self.model_class.model(**parameters)

        output = torch.argmax(output, dim=-1)[0].to("cpu")
        output = output.apply_(data_preparator.classes_ids.get)
        input_ids = parameters['input_ids'][0].to("cpu")
        final_ids = (input_ids * (output == 0) + output).tolist()

        original_tokens = data_preparator.tokenizer.convert_ids_to_tokens(
            final_ids, skip_special_tokens=True)
        sentence = data_preparator.tokenizer.convert_tokens_to_string(
            original_tokens)

        return sentence


TEST_DATA_PATH = "data/data_short.csv"
MODEL_NAME = "bert-base-german-cased"
PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict_multiple_masks_partially_wrong_all_articles.pt"
BATCH_SIZE = 16

bert_for_grammar_correction_evaluator = BertForGrammarCorrectionEvaluator(
    MODEL_NAME, PRETRAINED_MODEL_PATH)


sentence = "Eine Katze verfolgt eine Maus. Die Katze läuft durch einen Busch. Ein Mann hat die Maus gerettet. Der Mann kam aus einem Schuppen."
wrong_sentence = "Einer Katze verfolgt ein Maus. Die Katze läuft durch ein Busch. Ein Mann hat den Maus gerettet. Den Mann kam aus einen Schuppen."
prediction = bert_for_grammar_correction_evaluator.evaluate_on_sentences(
    wrong_sentence)

print("\nTRUE SENTENCE:\t\t", sentence)
print("\nWRONG SENTENCE:\t\t", wrong_sentence)
print("\nPREDICTED SENTENCE:\t", prediction, "\n")

sentence = "Das Kind eines Mannes hat eine Mütze in dem Bus vergessen. Der Busfahrer hat sie einer Lehrerin gegeben."
wrong_sentence = "Das Kind ein Mannes hat einer Mütze in den Bus vergessen. Den Busfahrer hat sie eine Lehrerin gegeben."
prediction = bert_for_grammar_correction_evaluator.evaluate_on_sentences(
    wrong_sentence)

print("\nTRUE SENTENCE:\t\t", sentence)
print("\nWRONG SENTENCE:\t\t", wrong_sentence)
print("\nPREDICTED SENTENCE:\t", prediction, "\n")
