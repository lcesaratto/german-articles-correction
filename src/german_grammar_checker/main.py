import torch
from transformers import AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

from german_grammar_checker.train import train_model_on_full_train_data
from german_grammar_checker.evaluate import evaluate_on_test_data


TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/eval.csv"
MODEL_NAME = "bert-base-german-cased"
BATCH_SIZE = 16
NUM_EPOCHS = 1

model, training_stats = train_model_on_full_train_data(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS)
print("\nTraining results: ", training_stats)

torch.save(model.state_dict(), "pretrained_model/model_state_dict.pt")
# torch.save(model, "unidad_09/pretrained_model/entire_model.pt")

testing_stats = evaluate_on_test_data(model, TEST_DATA_PATH, MODEL_NAME, BATCH_SIZE)
print("\nTesting results: ", testing_stats)