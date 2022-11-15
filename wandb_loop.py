import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb
from copy import deepcopy
#skal være en import
modelname = 'LSTM'
 
sweep_config = {
    'name': 'navn',
    'method': 'random', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_loss',
    'goal': 'minimize'  
        },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128, 256, 512]
        },
        'hidden_size': {
            'values': [8, 16, 32, 64, 128, 256, 512]
        },        
        'attention_heads': {
            'values': [1, 2, 4, 8, 16]
        },
        'encoding_size': {
            'values': [5, 10, 20, 40]
        },
        'optimizer': {
            'values': ['rAdam', 'adam', 'sgd', 'ranger']
        },
        'encoder_length': {
            'values': [6, 12, 24, 48, 92, 184, 268, 536]
        },
        'sequence_length': {
            'values': [6, 12, 24, 48, 92, 184, 268, 536]
        },
        'lr': {
            'values': [0.0001, 0.001, 0.01, 0.1]
        },
        'weight_decay': {
            'values': [0, 0.000001, 0.0001]
        },
        'dropout_rate': {
            'values': [0, 0.05, 0.1, 0.2]
        },
        'LSTM_layers': {
            'values': [1, 2, 4, 8, 16]
        },
        'n_encoder_layers': {
            'values': [1, 2, 4, 8, 16]
        },
        'n_decoder_layers': {
            'values': [1, 2, 4, 8, 16]
        },
        'days_training_length': {
            'values': [31, 62, 124, 248, 365, 540]
        },
    }
}
 
def set_hyp_config(modelname):
    sweep_config['name'] = modelname
    if modelname == "LSTM":
        print('speciel setting for: ', modelname)
        sweep_config['parameters']['batch_size']['values'] = [1] #LSTM tager kun batch_size på 1
        del sweep_config['parameters']['encoder_length'] #Fjerner alle transformers parametre som ikke er med i LSTM
        del sweep_config['parameters']['attention_heads']
        del sweep_config['parameters']['encoding_size']
        del sweep_config['parameters']['n_encoder_layers']
        del sweep_config['parameters']['n_decoder_layers']
        #Tænker der skal være noget til optim her, vil gætte på: sweep_config['parameters']['optimizer']['values'] = ['rAdam', 'adam']
 
    if modelname == "Queryselector" or modelname == "TFT":
        print('speciel setting for: ', modelname)
        del sweep_config['parameters']['LSTM_layers'] #Fjerner alle LSTM parametre
        del sweep_config['parameters']['sequence_length']
        #Tænker der skal være noget til optim her, vil gætte på: sweep_config['parameters']['optimizer']['values'] = ['sgd', 'ranger']
 
 
def wandb_initialize():
    sweep_id = wandb.sweep(sweep_config, project="Yggdrasil", entity="ygg_monkeys") #todo: dette laver en ny sweep.
    wandb.agent(sweep_id=sweep_id, function=sweep)
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)
 
def wandb_log(train_loss, val_loss, train_acc, val_acc):
    wandb.log({"train_loss": train_loss})
    wandb.log({"val_loss": val_loss})
    wandb.log({"train_acc": train_acc})
    wandb.log({"val_acc": val_acc})
 
def wandb_log_folds_avg(avg_val_acc, avg_val_loss):
    wandb.log({"avg_val_acc":avg_val_acc})
    wandb.log({"avg_val_loss":avg_val_loss})
 
def sweep():
    wandb.init(config=sweep_config)
    print(wandb.config.lr)
    run(wandb.config)
 
 
 
def get_data(dates, traininglength, df_features):
 
    #to get date as datetime object, so timedelta method can be used.
    start_date = datetime.fromisoformat(dates)
    #timedelta finds automatic the date x days before
    endate = start_date - timedelta(days= (traininglength+7))
 
    endindex = df_features.index[df_features['hour'] == dates][0]
    startindex = df_features.index[df_features['hour'] == str(endate)][0]
 
    df_season = df_features.iloc[startindex:endindex]
   
    return df_season
 
 
 
def run(hyper_dick):
 
        df_features = pd.read_csv(r"data\\dataset_dropNA.csv")
        season =  ["Winther", "Spring", "Summer", "Fall"]
        dates = ["2021-01-10 23:00:00", "2021-04-11 23:00:00", "2021-07-11 23:00:00", "2021-10-10 23:00:00" ]
        Total_average_mae_loss = 0
        Total_average_rmse_loss = 0

        print(hyper_dick.days_training_length)
        print(hyper_dick)
        for x in range(len(dates)):
            print(len(dates))
            df_season = get_data(dates[x], hyper_dick.days_training_length, df_features) #den her metode skal i lave
 
            if modelname == "LSTM":
                mae, rmse, predictions, target = model.train(df_season, hyper_dick)
            if modelname == "TFT":
                mae, rmse, predictions, target = model.train(df_season, hyper_dick)
            if modelname == "Queryselector":
                mae, rmse, predictions, target = model.train(df_season, hyper_dick)
            # train:
            #  - skal selv have early stopping
            #  - skal retunere mae, rmse, predictions for hver dag i ugen. (Husk at den skal loade den bedste)
            #       - mae, rmse: (1*7).
            #       - predictions: (36*7)
            average_mae_season = 0
            average_rmse_season = 0
            for i in range(len(mae)):
                wandb.log({f"{season[x]}_RMSE_loss": rmse[i]})
                wandb.log({f"{season[x]}_MAE_loss": mae[i]})
                average_mae_season +=mae
                average_rmse_season +=rmse
           
            wandb.log({f"{season[x]}_Average_MAE_Loss": (average_mae_season/7)})
            wandb.log({f"{season[x]}_Average_RMSE_Loss": (average_rmse_season/7)})
 
            Total_average_mae_loss += average_mae_season/7
            Total_average_rmse_loss += average_rmse_season/7
 
            notfirst15 = 15
            coomulator = 39
            for z in range((len(predictions))):
                if z == coomulator:
                    notfirst15 += coomulator
                    coomulator += coomulator
 
                if z > notfirst15:
                    wandb.log({f"{season[x]}_Predictions": predictions[z]})
                    wandb.log({f"{season[x]}_Target": target[z]})
       
        wandb.log({"Total_Average_MAE_Loss": Total_average_mae_loss/4})
        wandb.log({"Total_Average_RMSE_Loss": Total_average_rmse_loss/4})
 
modelnames = ["LSTM", "Queryselector", "TFT"]
modelname = modelnames[0]
set_hyp_config(modelname)
wandb_initialize()
#run("efwef") #modelnavn ++ antallet af dage vi træner/validere på

    