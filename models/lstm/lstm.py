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

    def train(self,df: pd.DataFrame, hyper_dick):
        self.model = None
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

        predictions = []
        MAEs = []
        RMSEs = []
        for i in range(7):
            df_features_loop = self.offset_dataframe(df_features,i)
            df_targets_loop = self.offset_dataframe(df_targets,i)

            x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_features_loop, df_targets_loop, sequence_length, batch_size, self.validation_length, self.validation_batchsize)

            x_train, y_train, x_test, y_test = self.convert_to_tensors(x_train, y_train, x_test, y_test)

            model_params = {'input_dim': x_train.size(dim=3),
                            'hidden_dim' : hidden_dim,
                            'layer_dim' : y_train.size(dim=3),
                            'output_dim' : 3,
                            'dropout_prob' : dropout}

            if i == 0 or self.model is None:
                self.create_new_model(model_params)

            optimizer = self.get_optimizer(hyperParams['optimizer'],learning_rate,weight_decay)
            loss_function = torch.nn.L1Loss()
            opt = Optimization(model=self.model, loss_fn=loss_function, optimizer=optimizer)

            if batch_size == 1:
                hn,cn = opt.train(train_features=x_train,targets=y_train,forward_hn_cn=True, plot_losses=False, model_path = "lstm_model.pt")
            else:
                hn,cn = opt.train(train_features=x_train,targets=y_train,forward_hn_cn=False, plot_losses=False, model_path = "lstm_model.pt")

            new_predictions = opt.evaluate(x_test,grab_last_batch(hn,layer_dim,hidden_dim),grab_last_batch(cn,layer_dim,hidden_dim))
            mae,rmse = get_mae_rmse(new_predictions,y_test)

            MAEs.append(mae)
            RMSEs.append(rmse)
            predictions.append(new_predictions)
        return MAEs, RMSEs, predictions


    def get_optimizer(self,optimizer_name,learning_rate,weight_decay):
        if optimizer_name == "rAdam":
            return optim.RAdam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            return optim.Adam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "ranger":
            raise Exception("Ranger optimizer is not supported yet")
            return Ranger
        else:
            raise Exception("Optimizer not found")

    def create_new_model(self,model_params):
        self.model = LSTMModel(**model_params)
        if torch.cuda.is_available():
            self.model.cuda()

    def offset_dataframe(self,df,i) -> pd.DataFrame:
        later_validation_hours = (6-i)*24
        unused_train_hours = i*24

        df_copy = df.drop(df.tail(later_validation_hours).index) # drop last n rows
        df_copy = df_copy.drop(df_copy.head(unused_train_hours).index) # drop first n rows
        return df_copy

    def convert_to_tensors(self,x_train, y_train, x_test, y_test):
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_test = x_test.cuda()
            #y_test = y_test.cuda()

        return x_train, y_train, x_test, y_test
            
        

