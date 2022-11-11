from turtle import color
from lstm_model import *
from lstm_train_test_splitter import lstm_train_test_splitter
import pandas as pd

def grab_last_batch(hn,dimension,hidden_size):
    return hn[:,-1,:].reshape(dimension,1,hidden_size)

def txt_to_list(file_dir: str):
    with open(file_dir) as f:
        return f.read().splitlines()

df_features = pd.read_csv(r"data\\dataset_dropNA.csv")

df_features = df_features[(df_features.index<np.percentile(df_features.index, 50))]

targets_cols = ['TH_NP15_GEN-APND','TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']
test_sequence_length = 38
x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_features, targets_cols, 730, 1, test_sequence_length, 1, normalize_features=True, remove_target_outliers=True)

if torch.cuda.is_available():
    x_train = torch.Tensor(x_train).cuda()
    y_train = torch.Tensor(y_train).cuda()
    x_test = torch.Tensor(x_test).cuda()
else:
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

import torch.optim as optim
input_dim = x_train.size(dim=3)
output_dim = y_train.size(dim=3)
hidden_dim = 256
layer_dim = 2
dropout = 0
n_epochs = 100
learning_rate = 0.001
weight_decay = 1e-6
model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = LSTMModel(**model_params).to(device)

loss_fn = nn.L1Loss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

hn,cn = opt.train(train_features=x_train,targets=y_train, n_epochs=n_epochs,forward_hn_cn=True,plot_losses=True, model_path = "lstm_model.pt")

predictions = opt.evaluate(x_test,grab_last_batch(hn,layer_dim,hidden_dim),grab_last_batch(cn,layer_dim,hidden_dim))
loss = opt.calculate_loss(predictions, y_test)
print(loss)

predictions = predictions.reshape(test_sequence_length,output_dim)
y_test = y_test.reshape(test_sequence_length,output_dim)
#plot predictions as dots and values as lines
plt.plot(y_test, label=targets_cols)
plt.plot(predictions, label=targets_cols, linestyle='dashed')
plt.legend()
plt.show()