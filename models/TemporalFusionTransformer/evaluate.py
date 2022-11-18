import torch
from pytorch_forecasting import TemporalFusionTransformer, Baseline
import copy
import sys
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from dataloader import get_time_varying_known_reals

from plot_predictions import plot_predictions

def get_best_tft(trainer) -> TemporalFusionTransformer:
    # hvis vi har trænet nu, så tager den den.
    # ellers så bbrug filen best_weights.ckpt
    # filen kan erstattes løbende af de bedste checkpoints
    best_model_path = trainer.checkpoint_callback.best_model_path
    #print("best model path:", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return best_tft

    #weights_path = 'best_weights_multi.ckpt'
    #try:
    #    path = best_model_path if best_model_path is not '' else weights_path
    #    best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    #except:
    #    # kører i debug:
    #    path = best_model_path if best_model_path is not '' else f"models/transformer/{weights_path}"
    #    best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    
    #return best_tft

def evaluate(trainer, val_dataloader):
    best_tft = get_best_tft(trainer)

    ys = []
    for x, (y, weight) in iter(val_dataloader):
        ys.append(y)
    targets = ys[0]

    predictions = best_tft.predict(val_dataloader)

    avg_mae, avg_rmse = get_mae_rmse(targets, predictions)

    preds_not_as_tensor = []
    targets_not_as_tensor = []
    for i in range(len(predictions)):
        preds_not_as_tensor.append(predictions[i].tolist())
        targets_not_as_tensor.append(targets[i].tolist())

    print("targets not as tensor:", targets_not_as_tensor)
    return preds_not_as_tensor, targets_not_as_tensor, avg_mae, avg_rmse

def get_mae_rmse(targets:list[torch.Tensor], predictions:list[torch.Tensor]) -> list[int]:
    """
    Tager targets og predictions for de 3 hubs og returnerer gennemsnitlig MAE og RMSE.
    """
    maes = []
    rmses = []
    for i in range(len(targets)):
        mean_abs_error = (targets[i] - predictions[i]).abs().mean()
        mean_squared_error = (targets[i] - predictions[i]).square().mean()
        root_mean_squared_error = mean_squared_error.sqrt()
        maes.append(mean_abs_error.item())
        rmses.append(root_mean_squared_error.item())
        
    avg_mae = sum(maes) / len(maes)
    avg_rmse = sum(rmses) / len(rmses)
    return avg_mae, avg_rmse

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

def predict_on_new_data(trainer):
    """
    encoder_data er den data som vi kan se tendenser i.
    Decoder data er vel bare vores forecasts i 6 timesteps uden hub values.
    """
    best_tft = get_best_tft(trainer)
    csv_path = "data/tmp_train.csv"
    try:
        data:pd.DataFrame = pd.read_csv(f'../../{csv_path}')
    except:
        data:pd.DataFrame = pd.read_csv(csv_path) # denne kører i debug mode.
    data['time_idx'] = range(0, len(data))
    data = data.drop(['hour'], axis=1)
    data['group'] = 0 # vi har kun 1 group, i.e california.

    print(data.head())

    # create dataset and loaders
    max_prediction_length = 6
    max_encoder_length = 24

    time_varying_known_reals = get_time_varying_known_reals(data)

    test_dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=[
            "TH_NP15_GEN-APND",
            "TH_SP15_GEN-APND",
            "TH_ZP26_GEN-APND"
        ],
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        #static_categoricals=["agency", "sku"],
        #static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        #time_varying_known_categoricals=["special_days", "month"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "TH_NP15_GEN-APND",
            "TH_SP15_GEN-APND",
            "TH_ZP26_GEN-APND"
        ],
        #target_normalizer=GroupNormalizer(
        #    groups=["agency", "sku"], transformation="softplus"
        #),  # use softplus and normalize by group
        #target_normalizer=MultiNormalizer(
        #[EncoderNormalizer(), TorchNormalizer()]
        #),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


    test_loader = test_dataset.to_dataloader(train=False, batch_size=6, num_workers=0)
    new_raw_predictions = best_tft.predict(test_loader, mode="prediction")
    print(new_raw_predictions)











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