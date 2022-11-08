import pandas as pd
from datetime import datetime

"""
Tilføjer "day", "month", "hours", "weekday" til et datasæt. (.csv)
"""

def get_datetime_object(row) -> datetime:
    datetime_str = row['hour'] #fx: 2022-09-01 01:00:00
    datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    return datetime_object

def label_day(row):
    date = get_datetime_object(row)
    return date.day

def label_month(row):
    date = get_datetime_object(row)
    return date.month

def label_hour(row):
    date = get_datetime_object(row)
    return date.hour

def label_weekday(row):
    date = get_datetime_object(row)
    return date.weekday()



def add_labels(csv_path = "data/dataset_dropNA.csv", output_name = 'datasetV3.csv'):
    try:
        data:pd.DataFrame = pd.read_csv(f'../{csv_path}')
    except:
        data:pd.DataFrame = pd.read_csv(csv_path) # denne kører i debug mode.

    data['month'] = data.apply (lambda row: label_month(row), axis=1)
    data['day'] = data.apply (lambda row: label_day(row), axis=1)
    data['weekday'] = data.apply (lambda row: label_weekday(row), axis=1)
    data['hours'] = data.apply (lambda row: label_hour(row), axis=1)

    print(data.head())

    data.to_csv(output_name, index=False)

add_labels()