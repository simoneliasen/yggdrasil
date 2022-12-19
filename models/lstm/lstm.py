import sys
sys.path.append('models/lstm/')

from lstm_model import LSTMModel, Optimization
from lstm_data_handler import *
import pandas as pd
import torch.optim as optim
import torch.cuda as cuda
#from pytorch_forecasting.optim import Ranger

class LSTM():
    def __init__(self) -> None:
        self.targets_cols = ['TH_NP15_GEN-APND','TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']
        self.validation_length = 39
        self.validation_batchsize = 1

    def train(self,df: pd.DataFrame, hyper_dict, list_of_indices):
        """
        Trains the model on 7 days and returns the loss of the best model
        Expected hyper_dict parameters:
        batch_size
        hidden_size       
        optimizer
        sequence_length
        lr
        weight_decay
        dropout_rate
        LSTM_layers
        days_training_length
        """
        df_features,df_targets = feature_label_split(df, self.targets_cols)
        df_features = df_features.drop(columns=['hour'], axis=1)

        #df_targets = remove_outliers(df_targets)
        df_features = normalize_dataframe(df_features)

        number_of_train_hours = len(df_features)-self.validation_length
        if number_of_train_hours*hyper_dict.batch_size < hyper_dict.sequence_length:
            raise Exception("Dataset is too short for the encoder length and batch size. Dataset length:{number_of_train_hours} encoder length:{sequence_length} batch size:{batch_size}")

        model_state_dict_path = "lstm_state_dict.pth"
        model_params = {'input_dim': len(df_features.columns), 'hidden_dim' : hyper_dict.hidden_size, 'layer_dim' : hyper_dict.LSTM_layers, 'output_dim' : 3, 'dropout_prob' : hyper_dict.dropout_rate}

        predictions = []
        targets = []
        MAEs = []
        RMSEs = []
        for i in range(len(list_of_indices)):
            #laver nu en ny model her i starten, skal nok lige dobbelt tjekkes
            print(f"")
            print(f"loading new model")
            self.create_new_model(model_params)
            torch.save(self.model.state_dict(), model_state_dict_path)
            print("Running" ,len(list_of_indices), "days")
            print(f"")
            print(f"Training for day {i}")
            #df_features_loop = self.offset_dataframe(df_features,i)
            #df_targets_loop = self.offset_dataframe(df_targets,i)
            df_features_loop = self.new_offset_dataframe(df_features,i,list_of_indices, 0)
            df_targets_loop = self.new_offset_dataframe(df_targets,i,list_of_indices, 0)
            #df_features_loop = self.offset_dataframe(df_features,i)
            #df_targets_loop = self.offset_dataframe(df_targets,i)

            x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_features_loop, df_targets_loop, hyper_dict.sequence_length, hyper_dict.batch_size, self.validation_length, self.validation_batchsize)
            x_train, y_train, x_test, y_test = self.convert_to_tensors(x_train, y_train, x_test, y_test)

            optimizer = self.get_optimizer(hyper_dict.optimizer,hyper_dict.lr,hyper_dict.weight_decay)
            loss_function = torch.nn.L1Loss()
            opt = Optimization(model=self.model, loss_fn=loss_function, optimizer=optimizer)

            if hyper_dict.batch_size > 1:
                forward_hidden_states = False
            else:
                forward_hidden_states = True

            hn,cn = opt.train(x_train,y_train,x_test,y_test,model_statedict_path=model_state_dict_path,forward_hn_cn=forward_hidden_states)
            print(f"")
            print("stepping and training for test, including validation day")
            df_features_loop = self.new_offset_dataframe(df_features,i,list_of_indices, 39)
            df_targets_loop = self.new_offset_dataframe(df_targets,i,list_of_indices, 39)
            hn,cn = opt.train_without_val(x_train,y_train,x_test,y_test,model_statedict_path=model_state_dict_path,forward_hn_cn=forward_hidden_states)
            
            new_predictions = opt.evaluate(x_test,grab_last_batch(hn,hyper_dict.LSTM_layers,hyper_dict.hidden_size),grab_last_batch(cn,hyper_dict.LSTM_layers,hyper_dict.hidden_size),model_state_dict_path)
            mae,rmse = get_mae_rmse(new_predictions,y_test)

            print(f"")
            print(f"Test for day {i} done")

            MAEs.append(mae)
            RMSEs.append(rmse)
            predictions.append(new_predictions)
            targets.append(y_test)
        return MAEs, RMSEs, predictions, targets


    def get_optimizer(self,optimizer_name,learning_rate,weight_decay):
        if optimizer_name == "rAdam":
            return optim.RAdam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            return optim.Adam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "ranger":
            raise Exception("Ranger optimizer is not supported yet")
            return None
        else:
            raise Exception("Optimizer not found")

    def create_new_model(self,model_params):
        self.model = LSTMModel(**model_params)
        if torch.cuda.is_available():
            self.model.cuda()

    def new_offset_dataframe(self,df,i,list_of_indices, daysoffset) -> pd.DataFrame:
        #daysoffset er bare step på 24 timer for at gå til næste dag mellem val og test
        end_index_of_test_or_val = list_of_indices[i]
        later_validation_hours = end_index_of_test_or_val + daysoffset #(slut punkt på test)
        unused_train_hours = end_index_of_test_or_val - 39 - 744 + daysoffset #(trækker træning og test timer fra slut på test, og finder derved start, og skal så slette alt inden)
        df_copy = df.iloc[unused_train_hours:later_validation_hours]
        #print(df_copy.head(5))
        #print(df_copy.tail(5))
        return df_copy

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