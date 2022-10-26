import numpy as np
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer
import pandas as pd

def get_train_val() -> list[TimeSeriesDataSet]:
    """
    Henter training og validation dataset.

    TODO: tager den al data med, eller skal jeg tilføje/fejrne til fx time_varying_known_reals?
    """
    csv_path = "data/hubs4.csv"
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

    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=["SP15_MERGED", "NP15_MERGED", "ZP26_MERGED"],
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        #static_categoricals=["agency", "sku"],
        #static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        #time_varying_known_categoricals=["special_days", "month"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "CAISO-SP15 Wind Power Generation Forecast", "CAISO Photovoltaic Power Generation Forecast"], # TODO: måske tilføj alle her?
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "SP15_MERGED",
            "NP15_MERGED",
            "ZP26_MERGED",
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

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True, allow_missing_timesteps=True)

    return [training, validation]