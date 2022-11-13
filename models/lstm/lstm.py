from lstm_model import LSTMModel, Optimization
from lstm_data_handler import *
import pandas as pd
import torch.optim as optim
import torch.cuda as cuda
from pytorch_forecasting.optim import Ranger

class lstm():
    def __init__(self) -> None:
        self.targets_cols = ['TH_NP15_GEN-APND','TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']
        self.validation_length = 36
        self.validation_batchsize = 1

    def train(self,df: pd.DataFrame, hyperParams:dict):
        
        df_features,df_targets = feature_label_split(df, self.targets_cols)
        
        df_targets = remove_outliers(df_targets)
        df_features = normalize_dataframe(df_features)
        
        learning_rate = hyperParams['learning_rate']
        dropout       = hyperParams['dropout']
        hidden_dim    = hyperParams['hidden_size']
        layer_dim     = hyperParams['LSTM_layers']
        weight_decay  = hyperParams['weight_decay']
        batch_size    = hyperParams['batch_size']
        sequence_length = hyperParams['encoder_length']
        
        number_of_train_hours = len(df_features)-self.validation_length
        if number_of_train_hours*batch_size < sequence_length:
            raise Exception("Dataset is too short for the encoder length and batch size. Dataset length:{number_of_train_hours} encoder length:{sequence_length} batch size:{batch_size}")
    
        for i in range(7):
            later_validation_hours = (6-i)*24
            unused_train_ours = i*24

            df_features_loop = df_features.drop(df.tail(later_validation_hours).index) # drop last n rows
            df_features_loop = df_features_loop.drop(df.head(unused_train_ours).index) # drop first n rows
            
            df_targets_loop = df_targets.drop(df.tail(later_validation_hours).index) # drop last n rows
            df_targets_loop = df_targets_loop.drop(df.head(unused_train_ours).index) # drop first n rows

            x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_features_loop, df_targets_loop, sequence_length, batch_size, self.validation_length, self.validation_batchsize)

            model_params = {'input_dim': x_train.size(dim=3),
                            'hidden_dim' : hidden_dim,
                            'layer_dim' : y_train.size(dim=3),
                            'output_dim' : 3,
                            'dropout_prob' : dropout}

            if i == 0 or self.model is None:
                if cuda.is_available():
                    self.model = LSTMModel(**model_params).cuda()
                else:
                    self.model = LSTMModel(**model_params)

            optimizer = self.get_optimizer(hyperParams['optimizer'],learning_rate,weight_decay)
            loss_function = torch.nn.L1Loss()
            opt = Optimization(model=self.model, loss_fn=loss_function, optimizer=optimizer)

            hn,cn = opt.train(train_features=x_train,targets=y_train, n_epochs=n_epochs,forward_hn_cn=True,plot_losses=True, model_path = "lstm_model.pt")

            predictions = opt.evaluate(x_test,grab_last_batch(hn,layer_dim,hidden_dim),grab_last_batch(cn,layer_dim,hidden_dim))
            loss = opt.calculate_loss(predictions, y_test)
            print(loss)

    def get_optimizer(self,optimizer_name,learning_rate,weight_decay):
        if optimizer_name == "rAdam":
            return optim.RAdam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            return optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "ranger":
            raise Exception("Ranger optimizer is not supported yet")
            return Ranger

        else:
            raise Exception("Optimizer not found")
