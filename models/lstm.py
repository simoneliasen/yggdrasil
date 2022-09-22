from random import shuffle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib as plt


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# load data
df = pd.read_csv("data/temperature2.csv")
df.drop(df.columns.difference(['hour','Alturas Temperature Forecast']), 1, inplace=True)
df.rename(columns={'Alturas Temperature Forecast':'target'}, inplace=True)


df['hour'] = pd.to_datetime(df['hour'], errors='coerce')

def one_hot_encode_datetime_column(df, column_name):
    df['day_of_week'] = df[column_name].dt.dayofweek
    df['month'] = df[column_name].dt.month
    df['day_of_month'] = df[column_name].dt.day
    df['week_of_year'] = df[column_name].dt.isocalendar().week
    df['hour_of_day'] = df[column_name].dt.hour
    #drop original datetime column
    df.drop(column_name, axis=1, inplace=True)
    #one-hot encode new columns
    df = pd.get_dummies(df, columns=['day_of_week','month','day_of_month','week_of_year','hour_of_day'])
    return df

df = one_hot_encode_datetime_column(df, 'hour')
print(df)

from sklearn.model_selection import train_test_split

def feature_label_split(df: pd.DataFrame, target_col: string):
    y = df[target_col]
    x = df.drop(columns=[target_col])
    return x, y

x,y = feature_label_split(df,'target')

def train_test_split(x: pd.DataFrame, y: pd.DataFrame, test_size: float):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = False)

from torch.utils.data import TensorDataset, DataLoader
batch_size = 64

train_features = torch.Tensor(x_train)
train_targets = torch.Tensor(y_train)
test_features = torch.Tensor(x_test)
test_targets = torch.Tensor(y_test)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()




model = LSTMModel()
criterion = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=0.8)


#train_input = torch.from_numpy(y[3:, :-1]) # (97, 999)
#train_target = torch.from_numpy(y[3:, 1:]) # (97, 999)
#test_input = torch.from_numpy(y[:3, :-1]) # (3, 999)
#test_target = torch.from_numpy(y[:3, 1:]) # (3, 999)


training_loop(4,model,optimiser,criterion,train_input,train_target,test_input,test_target)