# kraftigt inspireret af:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
from dataloader import get_train_val
from transformer import get_tft, get_trainer
from evaluate import evaluate, predict, predict_on_new_data

training, validation = get_train_val()
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

trainer = get_trainer()
tft = get_tft(training)

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

#predict_on_new_data(trainer)
evaluate(trainer, val_dataloader)
predict(trainer, val_dataloader)