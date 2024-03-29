from transformers import BertModel,\
    AdamW,\
    get_scheduler
import torch.nn as nn
import torch


class BertForGrammarCorrection(nn.Module):
    def __init__(self, model_name):
        super(BertForGrammarCorrection, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 25)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(output.last_hidden_state)
        output = self.linear2(output)
        return output


class Model:
    def __init__(self, model_name):
        self.model_name = model_name

    def init_model(self):
        self.model = BertForGrammarCorrection(self.model_name)

    def init_optimizer(self):
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=2e-5)

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def load_model_state_dict(self, pretrained_model_path):
        self.model.load_state_dict(torch.load(pretrained_model_path))

    def save_model_state_dict(self, pretrained_model_path):
        torch.save(self.model.state_dict(), pretrained_model_path)

    def init_scheduler(self, num_epochs, length_dataloader):
        self.num_training_steps = num_epochs * length_dataloader
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps
        )
