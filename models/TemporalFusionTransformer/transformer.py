import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from config_models import Config

def get_trainer() -> pl.Trainer:
    pl.seed_everything(42)
    # train model
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    #lr_logger = LearningRateMonitor()  # log the learning rate
    #logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=9999999,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        check_val_every_n_epoch=3,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],
    )
    return trainer

def get_tft(training, config:Config):
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.lr,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_heads,
        max_encoder_length=config.encoder_length,
        lstm_layers=config.LSTM_layers,
        dropout=config.dropout_rate,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        output_size=[39, 39, 39],  # 7 quantiles by default, 3 syv taller fordi vi har 3 hubs! (outputs)
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    return tft