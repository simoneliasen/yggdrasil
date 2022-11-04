from lstm_model import *
import pandas as pd
import math


def feature_label_split(df: pd.DataFrame, target_cols) -> tuple[pd.DataFrame,pd.DataFrame]:
    y = df[target_cols]
    x = df.drop(columns=target_cols)
    return x, y

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

def lstm_train_test_splitter(   df_features: pd.DataFrame, 
                                target_cols, 
                                sequence_length_train: int, 
                                batch_size_train: int, 
                                sequence_length_test: int = 24, 
                                batch_size_test: int = 1,
                                normalize_features: bool = True):
    """
    Splits the data into train and test sets. The test set will be the last
    test_size % of the data. The train set will be the remaining data.
    """
    df_features, df_targets = feature_label_split(df_features, target_cols)
    df_features.drop(columns=['hour'], axis=1, inplace=True)

    if normalize_features:
        df_features = df_features.apply(lambda x: x-x.mean())
        #df_features = df_features.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    test_features = df_features.tail(sequence_length_test * batch_size_test)
    test_targets = df_targets.tail(sequence_length_test * batch_size_test)
    train_features = df_features.head(len(df_features) - len(test_features))
    train_targets = df_targets.head(len(df_targets) - len(test_features))

    test_targets = test_targets.apply(lambda x: [150 if y > 150 else y for y in x])
    train_targets = train_targets.apply(lambda x: [150 if y > 150 else y for y in x])

    x_train = torch.Tensor(reshape_dataframe(train_features, sequence_length_train, batch_size_train))
    y_train = torch.Tensor(reshape_dataframe(train_targets, sequence_length_train, batch_size_train))
    x_test = torch.Tensor(reshape_dataframe(test_features, sequence_length_test, batch_size_test))
    y_test = torch.Tensor(reshape_dataframe(test_targets, sequence_length_test, batch_size_test))
    return(x_train, y_train, x_test, y_test)