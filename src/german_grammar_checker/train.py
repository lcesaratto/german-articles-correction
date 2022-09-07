import torch

from german_grammar_checker.model_preparation import Model
from german_grammar_checker.helper_functions import get_device
from german_grammar_checker.data_preparation import get_dataloaders


def train_model_on_full_train_data(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS):

    train_dataloader= get_dataloaders(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE)

    model_class = Model(MODEL_NAME, NUM_EPOCHS, len(train_dataloader))
    model, optimizer, lr_scheduler = model_class.get_model_optimizer_scheduler()

    device = get_device()
    model = model.to(device)

    training_stats = []

    try:
        for epoch in range(NUM_EPOCHS):
            print(f"EPOCH {epoch+1}/{NUM_EPOCHS}\n")
            model.train()
            total_train_loss = 0

            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                parameters = {
                    "input_ids" : batch[0].to(device),
                    "attention_mask" :  batch[1].to(device), 
                    "labels" : batch[2].to(device)
                }
                outputs = model(**parameters)

                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % 100 == 0 and step != 0:
                    print(f"BATCH {step}/{len(train_dataloader)}:\tTraining loss({loss.item()})")

            training_stats.append({
                "epoch":epoch+1,
                "training_loss":total_train_loss/len(train_dataloader)
                })

            print(f"\nAvg training loss:    {training_stats[epoch]['training_loss']}")

    except RuntimeError as e:
        print(e)
    
    return model, training_stats