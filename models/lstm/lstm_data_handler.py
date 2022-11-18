from pyparsing import remove_quotes
from lstm_model import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt 
def grab_last_batch(hn,dimension,hidden_size):
    return hn[:,-1,:].reshape(dimension,1,hidden_size)

def feature_label_split(df: pd.DataFrame, target_cols):
    y = df[target_cols]
    x = df.drop(columns=target_cols)
    return x, y

def get_mae_rmse(targets:list[torch.Tensor], predictions:list[torch.Tensor]):
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

def calculate_excess_rows(df, sequence_length, batch_size):
    number_of_batches = len(df) / (sequence_length * batch_size)
    number_of_batches = math.floor(number_of_batches)
    rows_to_drop = len(df) - (sequence_length * batch_size * number_of_batches)
    return rows_to_drop, number_of_batches

def reshape_dataframe(df, sequence_length, batch_size):
    rows_to_drop,number_of_batches = calculate_excess_rows(df, sequence_length, batch_size)
    df = df.iloc[rows_to_drop:, :]
    data = np.array(df.values)    
    data = data.reshape(number_of_batches, batch_size, sequence_length, df.shape[1])
    return data

def remove_outliers(df):
    df = df.apply(lambda x: [np.nan if y > 80 or y < -10 else y for y in x])
    df.interpolate(method='spline', order=1, limit=10, limit_direction='both',inplace=True)
    return df

def normalize_dataframe(df):
    return df.apply(lambda x: x-x.mean())

def differenciate_dataframe(df):
    return df.diff()

def lstm_train_test_splitter(   df_features: pd.DataFrame, 
                                df_targets: pd.DataFrame, 
                                sequence_length_train: int, 
                                batch_size_train: int, 
                                sequence_length_test: int = 24, 
                                batch_size_test: int = 1):
    """
    Splits the data into train and test sets. The test set will be the last
    test_size % of the data. The train set will be the remaining data.
    """    

    test_features = df_features.tail(sequence_length_test * batch_size_test)
    test_targets = df_targets.tail(sequence_length_test * batch_size_test)
    train_features = df_features.head(len(df_features) - len(test_features))
    train_targets = df_targets.head(len(df_targets) - len(test_features))

    x_train = reshape_dataframe(train_features, sequence_length_train, batch_size_train)
    y_train = reshape_dataframe(train_targets, sequence_length_train, batch_size_train)
    x_test = reshape_dataframe(test_features, sequence_length_test, batch_size_test)
    y_test = reshape_dataframe(test_targets, sequence_length_test, batch_size_test)
    return(x_train, y_train, x_test, y_test)