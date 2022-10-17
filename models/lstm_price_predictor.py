from lstm_model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from collections.abc import Iterable

def txt_to_list(file_dir: str):
    with open(file_dir) as f:
        return f.read().splitlines()


df = pd.read_csv(r"data\dataset.csv")

df.drop(columns=['hour'],axis=1,inplace=True)
#df.rename(columns={'Alturas Temperature Forecast':'target'}, inplace=True)
df.dropna(inplace=True)

print(df['CAISO-SP15 Photovoltaic power Generation Forecast'])

bad_columns = txt_to_list(r"data\caiso_columns.txt")
df.drop(columns=bad_columns,inplace=True)

#read a txt file with the names of the columns to be used as targets
with open(r"data\pricepoints.txt") as f:
    targets = f.read().splitlines()


def feature_label_split(df: pd.DataFrame, target_cols):
    y = df[target_cols]
    x = df.drop(columns=target_cols)
    return x, y

x, y = feature_label_split(df, targets)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

print(x_train.columns)
print(x_train['BPA Total Hydro Power Generation Forecast'])
batch_size = 64
print(x_train.values[0])
train_features = torch.Tensor(x_train.values)
train_targets = torch.Tensor(y_train.values)
test_features = torch.Tensor(x_test.values)
test_targets = torch.Tensor(y_test.values)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

"""
import torch.optim as optim
input_dim = len(x_train.columns)
output_dim = len(y_train.columns)
hidden_dim = 64
layer_dim = 1
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

#predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)

#plot predictions as dots and values as lines
plt.plot(values, label='values')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
"""