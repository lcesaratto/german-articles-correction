from transformers import    BertModel,\
                            AdamW,\
                            get_scheduler
import torch.nn as nn
import torch

class BertForGrammarCorrection(nn.Module):
    def __init__(self, model_name):
        super(BertForGrammarCorrection, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 768)
    
    def forward(self, ids, mask):
        output = self.bert(input_ids=ids, attention_mask=mask)
        print(output)
        print(output[0].shape)
        exit()
        output = self.linear(output[0].view(-1,768))
        return output

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def init_model_and_optimizer(self):
        self.model = BertForGrammarCorrection(self.model_name)
        self.optimizer = AdamW(self.model.parameters(),lr = 2e-5)

    def load_pretrained_model(self, pretrained_model_path):
        self.model = BertForGrammarCorrection(self.model_name)
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
    
    def get_model_optimizer_scheduler(self):
        return self.model, self.optimizer, self.lr_scheduler

    def get_num_training_steps(self):
        return self.num_training_steps