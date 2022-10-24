import torch
from pytorch_forecasting import TemporalFusionTransformer

def get_best_tft(trainer):
    # hvis vi har trænet nu, så tager den den.
    # ellers så bbrug filen best_weights.ckpt
    # filen kan erstattes løbende af de bedste checkpoints
    best_model_path = trainer.checkpoint_callback.best_model_path
    path = best_model_path if best_model_path is not '' else 'best_weights.ckpt'
    best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    return best_tft

def evaluate(trainer, val_dataloader):
    best_tft = get_best_tft(trainer)

    # calcualte mean absolute error on validation set
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)
    mean_abs_error = (actuals - predictions).abs().mean()
    print("mean abs error:", mean_abs_error)


def predict(trainer, val_dataloader):
    best_tft = get_best_tft(trainer)
    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    for idx in range(2):  # plot 10 examples
        plot = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        plot.savefig(f'my_plot{idx}.png')