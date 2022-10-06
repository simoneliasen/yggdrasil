from asyncio.windows_events import NULL
from tkinter.ttk import Style
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from random import shuffle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.values_assigned = False

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
        return out[:,-1]


# load data
df = pd.read_csv("data/temperature2.csv")

df.drop(columns=df.columns.difference(['hour','Alturas Temperature Forecast']),axis=1,inplace=True)
df.rename(columns={'Alturas Temperature Forecast':'target'}, inplace=True)
df.dropna(inplace=True)

df['hour'] = pd.to_datetime(df['hour'], errors='coerce')


def one_hot_encode_datetime_column(df, column_name):
    df['day_of_week'] = df[column_name].dt.dayofweek
    df['month'] = df[column_name].dt.month
    df['day_of_month'] = df[column_name].dt.day
    df['week_of_year'] = df[column_name].dt.isocalendar().week
    df['hour_of_day'] = df[column_name].dt.hour
    # drop original datetime column
    df.drop(column_name, axis=1, inplace=True)
    # one-hot encode new columns
    df = pd.get_dummies(df, columns=[
                        'day_of_week', 'month', 'day_of_month', 'week_of_year', 'hour_of_day'])
    return df


df = one_hot_encode_datetime_column(df, 'hour')
print(df.columns.size)

def feature_label_split(df: pd.DataFrame, target_col: string):
    y = df[target_col]
    x = df.drop(columns=[target_col])
    return x, y


x, y = feature_label_split(df, 'target')



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

batch_size = 64

train_features = torch.Tensor(x_train.values)
train_targets = torch.Tensor(y_train.values)
test_features = torch.Tensor(x_test.values)
test_targets = torch.Tensor(y_test.values)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

import datetime as datetime
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

    def train(self, train_loader: DataLoader, batch_size = 64, n_epochs:int = 50, n_features:int =1):
        model_path = f'models/test_lstm'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                #print(y_batch)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            """
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            """

            if (epoch <= n_epochs) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"# Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())
        return predictions, values


    def plot_losses(self):
        
        plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


import torch.optim as optim

input_dim = len(x_train.columns)
output_dim = 1
hidden_dim = 64
layer_dim = 2
batch_size = 64
dropout = 0
n_epochs = 100
learning_rate = 1e-1
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = LSTMModel(**model_params).to(device)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)

#plot predictions as dots and values as lines
plt.plot(values, label='values')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()