import numpy as np
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer
import pandas as pd
from config_models import Config

def get_train_val(data:pd.DataFrame, weekday:int, config:Config):

    """
    Henter training og validation dataset.
    """
    data = data.dropna()
    data['time_idx'] = range(0, len(data))
    data = data.drop(['hour'], axis=1)
    data['group'] = 0 # vi har kun 1 group, i.e california.

    #print(data.head())
    # create dataset and loaders
    max_prediction_length = 39

    # vælg dag til validering: altså til en rykke en dag
    #print("full data shape:", data.shape)
    end_weekday_offset = (6 * 24) - weekday * 24 # e.g. for mandags prediction offsetter vi med 6 dage fra søndag. og for søndag offsetter vi 0 dage.
    end_weekday_cutoff = data["time_idx"].max() - end_weekday_offset
    start_weekday_cutoff = weekday * 24
    data = data[lambda x: x.time_idx <= end_weekday_cutoff] # vælger hvor data skal slutte
    data = data[lambda x: x.time_idx >= start_weekday_cutoff] # vælger hvor data skal starte
    print("weekday", weekday, "shape:", data.shape)

    time_varying_known_reals = get_time_varying_known_reals(data)
    training_cutoff = data["time_idx"].max() - max_prediction_length

    #måned, dag osv. skal åbenbart være strings:
    data['month'] = data['month'].astype(str)
    data['day'] = data['day'].astype(str)
    data['weekday'] = data['weekday'].astype(str)
    data['hours'] = data['hours'].astype(str)


    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=[
            "TH_NP15_GEN-APND",
            "TH_SP15_GEN-APND",
            "TH_ZP26_GEN-APND"
        ],
        group_ids=["group"],
        min_encoder_length=config.encoder_length,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=config.encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        #static_categoricals=["agency", "sku"],
        #static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        time_varying_known_categoricals=["month", "day", "weekday", "hours"],
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

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True, allow_missing_timesteps=True)

    return [training, validation]


def get_time_varying_known_reals(data:pd.DataFrame):
    """
    Her henter vi bare alle vores navne på forecasts i et array.
     - Dvs. alle kolonner i csv'en efter hubsne.
    """
    last_hub_column_index = data.columns.get_loc('TH_ZP26_GEN-APND')
    #print(data.info())
    rest_columns = data.iloc[:,last_hub_column_index+1:]
    column_names = list(rest_columns.columns)
    column_names.remove("group")

    column_names.remove("month")
    column_names.remove("day")
    column_names.remove("weekday")
    column_names.remove("hours")
    # tilføj evt. flere her, hvis der er nye features der ikke er kendt på forhånd.
    return column_names