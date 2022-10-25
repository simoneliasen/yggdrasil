import torch
from pytorch_forecasting import TemporalFusionTransformer, Baseline
import copy

import sys
sys.path.append('../../')
from utils.plot_predictions import plot_predictions

def get_best_tft(trainer):
    # hvis vi har trænet nu, så tager den den.
    # ellers så bbrug filen best_weights.ckpt
    # filen kan erstattes løbende af de bedste checkpoints
    best_model_path = trainer.checkpoint_callback.best_model_path

    weights_path = 'best_weights_multi.ckpt'
    try:
        path = best_model_path if best_model_path is not '' else weights_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    except:
        # kører i debug:
        path = best_model_path if best_model_path is not '' else f"models/transformer/{weights_path}"
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
    copy_loader = copy.deepcopy(val_dataloader)

    ys = []
    for x, (y, weight) in iter(copy_loader):
        ys.append(y)
    targets = ys[0]

    predictions = best_tft.predict(val_dataloader)

    plot_predictions(predictions, targets)
    return predictions










def print_benchmarks(best_tft, val_dataloader):
    """
    VIRKEDE KUN MED 1 TARGET - SKAL RETTES TIL!
    """
    print_benchmark("best_tft", best_tft, val_dataloader)
    print_benchmark("Baseline", Baseline(), val_dataloader)

def print_benchmark(model_str, model, val_dataloader):
    """
    udregner en simpel score så vi kan sammenligne performance.
    """
    copy_loader = copy.deepcopy(val_dataloader)
    actuals = torch.cat([y for x, (y, weight) in iter(copy_loader)])
    predictions = model.predict(copy_loader)
    benchmark = (actuals - predictions).abs().mean().item()
    print(f"{model_str} score: {benchmark}")