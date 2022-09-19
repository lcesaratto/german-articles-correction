from german_grammar_checker.grammar_correction.data_preparation import DataPreparator
from german_grammar_checker.grammar_correction.helper_functions import get_device
from german_grammar_checker.grammar_correction.model_preparation import Model
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


class BertForGrammarCorrectionTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_class = Model(model_name)
        self.model_class.init_model()
        self.model_class.init_optimizer()
        self.model_class.init_criterion()

        self.device = get_device()
        self.model_class.model.to(self.device)

    def train_model_on_full_train_data(self, batch_size, num_epochs, train_data_path, one_mask=True):
        data_preparator = DataPreparator(self.model_name, batch_size)
        train_dataloader = data_preparator.get_dataloaders(
            train_data_path, one_mask)

        self.model_class.init_scheduler(num_epochs, len(train_dataloader))

        training_stats = {"epoch": [], "step": [], "training_loss": []}

        print("\nBegining training...")
        for epoch in range(num_epochs):
            print(f"EPOCH {epoch+1}/{num_epochs}")
            self.model_class.model.train()

            for step, batch in enumerate(train_dataloader):
                self.model_class.model.zero_grad()
                parameters = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask":  batch[1].to(self.device),
                }
                output = self.model_class.model(
                    **parameters)

                loss = self.model_class.criterion(
                    output.transpose(1, 2),
                    batch[2].to(self.device))
                loss.backward()

                self.model_class.optimizer.step()
                self.model_class.lr_scheduler.step()
                self.model_class.optimizer.zero_grad()

                if step % 100 == 0 and step != 0:
                    print(
                        f"BATCH {step}/{len(train_dataloader)}:\tTraining loss({loss.item()})")

                    training_stats["epoch"].append(epoch+1)
                    training_stats["step"].append(step+1)
                    training_stats["training_loss"].append(loss.item())

        return training_stats

    def save_model_state_dict(self, pretrained_model_path):
        self.model_class.save_model_state_dict(pretrained_model_path)


# # ONE MASK PER SENTENCE
# TRAIN_DATA_PATH = "data/data_short_one_mask_per_sentence.csv"
# MODEL_NAME = "bert-base-german-cased"
# BATCH_SIZE = 16
# NUM_EPOCHS = 1
# PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict_single_mask.pt"

# bert_for_grammar_correction_trainer = BertForGrammarCorrectionTrainer(
#     MODEL_NAME)

# training_stats = bert_for_grammar_correction_trainer.train_model_on_full_train_data(
#     BATCH_SIZE, NUM_EPOCHS, TRAIN_DATA_PATH)
# training_stats = pd.DataFrame(training_stats)
# training_stats.to_csv("pretrained_model/training_stats_single_mask.csv", index=False)

# bert_for_grammar_correction_trainer.save_model_state_dict(
#     PRETRAINED_MODEL_PATH)


# # MULTIPLE MASKS PER SENTENCE
# TRAIN_DATA_PATH = "data/data_short_multiple_masks_per_sentence.csv"
# MODEL_NAME = "bert-base-german-cased"
# BATCH_SIZE = 16
# NUM_EPOCHS = 1
# PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict_multiple_masks.pt"
# ONE_MASK = False

# bert_for_grammar_correction_trainer = BertForGrammarCorrectionTrainer(
#     MODEL_NAME)

# training_stats = bert_for_grammar_correction_trainer.train_model_on_full_train_data(
#     BATCH_SIZE, NUM_EPOCHS, TRAIN_DATA_PATH, ONE_MASK)
# training_stats = pd.DataFrame(training_stats)
# training_stats.to_csv(
#     "pretrained_model/training_stats_multiple_masks.csv", index=False)

# bert_for_grammar_correction_trainer.save_model_state_dict(
#     PRETRAINED_MODEL_PATH)


# MULTIPLE MASKS PER PARTIALLY WRONG SENTENCE
TRAIN_DATA_PATH = "data/data_short_multiple_masks_per_sentence_partially_wrong.csv"
MODEL_NAME = "bert-base-german-cased"
BATCH_SIZE = 16
NUM_EPOCHS = 1
PRETRAINED_MODEL_PATH = "pretrained_model/model_state_dict_multiple_masks_partially_wrong.pt"
ONE_MASK = False

bert_for_grammar_correction_trainer = BertForGrammarCorrectionTrainer(
    MODEL_NAME)

training_stats = bert_for_grammar_correction_trainer.train_model_on_full_train_data(
    BATCH_SIZE, NUM_EPOCHS, TRAIN_DATA_PATH, ONE_MASK)
training_stats = pd.DataFrame(training_stats)
training_stats.to_csv(
    "pretrained_model/training_stats_multiple_masks_partially_wrong.csv", index=False)

bert_for_grammar_correction_trainer.save_model_state_dict(
    PRETRAINED_MODEL_PATH)
