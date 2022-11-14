# kraftigt inspireret af:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
from dataloader import get_train_val
from transformer import get_tft, get_trainer
from evaluate import evaluate, predict, predict_on_new_data, get_best_tft
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
import pytorch_lightning as pl

class TFT:
    def train(self, data:pd.DataFrame):
        tft:TemporalFusionTransformer
        trainer:pl.Trainer = get_trainer()

        predictions = []
        MAEs = []
        RMSEs = []
        for weekday in range(2):
            training, validation = get_train_val(data, weekday)
            batch_size = 128  # set this between 32 to 128
            train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

            tft = get_tft(training) if weekday == 0 else get_best_tft(trainer)

            # fit network
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            preds, avg_mae, avg_rmse = evaluate(trainer, val_dataloader)
            predictions.append(preds)
            MAEs.append(avg_mae)
            RMSEs.append(avg_rmse)

        print("predictions final:", predictions)
        print("mae final:", MAEs)
        print("rmse final:", RMSEs)
        return MAEs, RMSEs, predictions

    def debug(self):
        csv_path = "data/datasetV3.csv"
        try:
            data:pd.DataFrame = pd.read_csv(f'../../{csv_path}')
        except:
            data:pd.DataFrame = pd.read_csv(csv_path) # denne kører i debug mode

        data = data.head(1000)
        print(data)
        self.train(data)

test = TFT()
test.debug()