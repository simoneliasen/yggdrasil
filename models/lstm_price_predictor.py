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
#df_features[df_columns] = scaler.fit_transform(df_features[df_columns])
df_features = df_features.apply(lambda x: x-x.mean())
#df_features = (df_features - df_features.mean()) / df_features.std()

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=False)

print(x_train.shape)
print(x_test.shape)
batch_size = 2
train_features = torch.Tensor(x_train.values)
train_targets = torch.Tensor(y_train.values)
test_features = torch.Tensor(x_test.values)
test_targets = torch.Tensor(y_test.values)

print(train_features.size())

import torch.optim as optim
input_dim = len(x_train.columns)
output_dim = len(y_train.columns)
hidden_dim = len(x_train.columns)
layer_dim = 1
batch_size = batch_size
dropout = 0
n_epochs = 200
learning_rate = 0.5
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
#h0,c0 = opt.train(train_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
hn,cn = opt.train(train_features,targets=train_targets, n_epochs=n_epochs)
#opt.plot_losses()
print(hn.detach().numpy())
print(cn.detach().numpy())


predictions = opt.evaluate(test_features, h0=hn, c0=cn)

#plot predictions as dots and values as lines
plt.plot(y_test.values, label='values')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
