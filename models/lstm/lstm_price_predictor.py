from lstm_model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from sklearn.preprocessing import MinMaxScaler


def txt_to_list(file_dir: str):
    with open(file_dir) as f:
        return f.read().splitlines()


df_features = pd.read_csv(r"data\dataset_new.csv")

faulty_columns = txt_to_list(r"data\faulty_columns.txt")
df_features.drop(columns=faulty_columns,inplace=True)

not_used_pricepoints = txt_to_list(r"data\pricepoints.txt")
df_features.drop(columns=not_used_pricepoints,inplace=True)

df_features.drop(columns=['hour'],axis=1,inplace=True)

df_features.interpolate(method='spline', order=1, limit=10, limit_direction='both',inplace=True)
scaler = MinMaxScaler()
df_columns = df_features.columns

df_features = df_features.apply(lambda x: x-x.mean())

#df.dropna(inplace=True)

df_targets = pd.read_csv(r"data\hubs.csv")
df_targets.drop(columns=['hour'],inplace=True)

def feature_label_split(df: pd.DataFrame, target_cols):
    y = df[target_cols]
    x = df.drop(columns=target_cols)
    return x, y

#x, y = feature_label_split(df, targets)
x = df_features
y = df_targets

sequence_length = 12
batch_size = 2

number_of_rows = len(x.index)
number_of_batches = number_of_rows / (sequence_length * batch_size)

import math
number_of_batches = math.floor(number_of_batches)

rows_to_drop = number_of_rows - (sequence_length * batch_size * number_of_batches)
x = np.array(x.iloc[rows_to_drop: , :].values)
y = np.array(y.iloc[rows_to_drop: , :].values)
x_batches = x.reshape(number_of_batches, batch_size, sequence_length, x.shape[1])
y_batches = y.reshape(number_of_batches, batch_size, sequence_length, y.shape[1])


#TODO Update this to be not just the last batch, but some other thing
x_train = x_batches[:-1]
y_train = y_batches[:-1]


x_test = x_batches[-1:]
y_test = y_batches[-1:]

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=False)

train_features = torch.Tensor(x_train)
train_targets = torch.Tensor(y_train)
test_features = torch.Tensor(x_test)
test_targets = torch.Tensor(y_test)

import torch.optim as optim
input_dim = len(df_features.columns)
output_dim = len(df_targets.columns)
hidden_dim = 256
layer_dim = 2
sequence_length = sequence_length
dropout = 0
n_epochs = 100
learning_rate = 0.3
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = LSTMModel(**model_params).to(device)

loss_fn = nn.L1Loss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

hn,cn = opt.train(train_features,targets=train_targets, n_epochs=n_epochs,forward_hn_cn=False)

predictions = opt.evaluate(test_features,hn,cn)
loss = opt.calculate_loss(predictions, test_targets)
print(loss)

predictions = predictions.reshape(batch_size*sequence_length,output_dim)
y_test = y_test.reshape(batch_size*sequence_length,output_dim)
#plot predictions as dots and values as lines
plt.plot(y_test, label='values')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()