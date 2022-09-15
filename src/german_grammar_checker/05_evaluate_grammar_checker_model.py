from german_grammar_checker.data_preparation import DataPreparator
from german_grammar_checker.helper_functions import get_device
from german_grammar_checker.model_preparation import Model
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


TEST_DATA_PATH = "data/data_short.csv"
MODEL_NAME = "bert-base-german-cased"
PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict.pt"
BATCH_SIZE = 16


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


bert_for_grammar_correction_evaluator = BertForGrammarCorrectionEvaluator(
    MODEL_NAME, PRETRAINED_MODEL_PATH)
# testing_stats = bert_for_grammar_correction_evaluator.evaluate_on_test_data(
#     BATCH_SIZE, TEST_DATA_PATH)
# testing_stats = pd.DataFrame(testing_stats)
# testing_stats.to_csv("testing_stats.csv", index=False)

sentence = "Das Baby schreit. Die Mutter gibt dem Baby den Schnuller und nimmt es in den Arm. Dann geht die Mutter mit dem Baby auf dem Arm in die Küche."
wrong_sentence = "Der Baby schreit. Der Mutter gibt den Baby der Schnuller und nimmt es in den Arm. Dann geht der Mutter mit den Baby auf den Arm in der Küche."
prediction = bert_for_grammar_correction_evaluator.evaluate_on_sentences(
    wrong_sentence)

print("\nTRUE SENTENCE:\t\t", sentence)
print("\nWRONG SENTENCE:\t\t", wrong_sentence)
print("\nPREDICTED SENTENCE:\t", prediction, "\n")
