# kraftigt inspireret af:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
import sys
import traceback

try:
    from dataloader import get_train_val
except:
    # setting path
    try:
        sys.path.append('/content/yggdrasil/models/TemporalFusionTransformer/')
        from dataloader import get_train_val
    except:
        sys.path.append('models/TemporalFusionTransformer/')
        from dataloader import get_train_val
from transformer import get_tft, get_trainer
from evaluate import evaluate, predict, predict_on_new_data, get_best_tft
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
import pytorch_lightning as pl
from config_models import Config
import time

class TFT:
    def train(data:pd.DataFrame, config:Config):
        return TFT.train2(data, config)

        # det her nede er godt til debug, da det giver bedre info end wandb.
        try:
            return TFT.train2(data, config)
        except Exception as e:
            # exit gracefully, so wandb logs the problem
            print("Exception:", e)
            print(traceback.print_exc(), file=sys.stderr)
            time.sleep(10)
            exit(1)

    def train2(data:pd.DataFrame, config:Config):
        tft:TemporalFusionTransformer
        trainer:pl.Trainer = get_trainer()
        predictions = []
        targets = []
        MAEs = []
        RMSEs = []
        for weekday in range(7):
            training, validation = get_train_val(data, weekday)
            batch_size = config.batch_size  # set this between 32 to 128
            train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

            tft = get_tft(training, config) if weekday == 0 else get_best_tft(trainer)

            # fit network
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            preds, tgts, avg_mae, avg_rmse = evaluate(trainer, val_dataloader)
            predictions.append(preds)
            targets.append(tgts)
            MAEs.append(avg_mae)
            RMSEs.append(avg_rmse)
            print("MAEs:", MAEs)

        print("predictions final:", predictions)
        print("mae final:", MAEs)
        print("rmse final:", RMSEs)
        return MAEs, RMSEs, predictions, targets

    def debug(self):
        csv_path = "data/datasetV3.csv"
        try:
            data:pd.DataFrame = pd.read_csv(f'../../{csv_path}')
        except:
            data:pd.DataFrame = pd.read_csv(csv_path) # denne kører i debug mode

        data = data.head(1000)
        print(data)
        self.train(data)