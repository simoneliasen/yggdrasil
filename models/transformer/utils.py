import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import math
import numpy as np

def get_dataloaders(dataset:str = "../../data/dataset.csv", valPercentage:int = 20, testPercentage:int = 20):
    """
    Loader csv filen og returnerer train,val,test-dataloaders. (numpy arrays)
    """
    df = pd.read_csv(dataset)
    df = df.drop(["hour"], axis=1) # fjern hour colonne: (e.g. 2022-07-31 01:00:00)
    num_rows = len(df.index)
    valIdx = math.floor(num_rows * ((100-(valPercentage + testPercentage))/100))
    testIdx = math.floor(num_rows * ((100-(testPercentage))/100))

    #df.iloc er bare: "fra række_x til række_y"
    train_df = df.iloc[:valIdx,:]
    val_df = df.iloc[valIdx:testIdx,:]
    test_df = df.iloc[testIdx:,:]

    print("Train, val, test batches:")
    train_dataloader = batchify_df(train_df)
    val_dataloader = batchify_df(val_df)
    test_dataloader = batchify_df(test_df)

    return [train_dataloader, val_dataloader, test_dataloader]

def batchify_df(df:pd.DataFrame, batch_size:int=24):
    """
    Input: En dataframe + batch_size (antal timer)
    Output: loader
    """
    batches = []
    for idx in range(0, len(df.index), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(df.index):
            batch = df.iloc[idx:idx+batch_size,:]
            batches.append(batch.to_numpy())

    print(f" - {len(batches)} batches of size {batch_size}")

    return batches