import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb
from copy import deepcopy
from models.TemporalFusionTransformer.index import TFT
from models.lstm.lstm import LSTM
#skal være en import

 
sweep_config = {
    'name': 'navn',
    'method': 'random', #grid, random, bayesian
    'metric': {
    'name': 'Total_Average_MAE_Loss',
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
            'values': [31, 62, 124, 248, 365, 520]
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
        sweep_config['parameters']['optimizer']['values'] = ['rAdam', 'adam', 'sgd']
        #Tænker der skal være noget til optim her, vil gætte på: sweep_config['parameters']['optimizer']['values'] = ['rAdam', 'adam']
 
    if modelname == "Queryselector" or modelname == "TFT":
        print('speciel setting for: ', modelname)
        del sweep_config['parameters']['sequence_length']
        #Tænker der skal være noget til optim her, vil gætte på: sweep_config['parameters']['optimizer']['values'] = ['sgd', 'ranger']
 
 
def wandb_initialize(modelname):
    #sweep_id = wandb.sweep(sweep_config, project="Yggdrasil", entity="ygg_monkeys") #todo: dette laver en ny sweep.
    sweep_ids = {
        "LSTM":"ywympjpq", 
        "Queryselector":"0srx7ptw", 
        "TFT":"29snzczn"
    }
    wandb.agent(sweep_id=sweep_ids[modelname], function=sweep, project="Yggdrasil", entity="ygg_monkeys")
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)
 
def sweep():
    wandb.init(config=sweep_config)
    run(wandb.config)
 
 
 
def get_data(dates, traininglength, df_features):
    
    #to get date as datetime object, so timedelta method can be used.
    start_date = datetime.fromisoformat(dates)
    #timedelta finds automatic the date x days before 
    endindex = df_features.index[df_features['hour'] == dates][0]
    
    i = 7
    timestep_exist = False
    #til at håndtere manglende timesteps i csv
    while timestep_exist is False:
        endate = start_date - timedelta(days= (traininglength+i))
        timestep_exist = timestep_check(df_features, endate)
        i += 1
        
    startindex = df_features.index[df_features['hour'] == str(endate)][0]
 
    df_season = df_features.iloc[startindex:endindex]
   
    return df_season
 
def timestep_check(df_features, endate):
    xx = True
    try:
        df_features.index[df_features['hour'] == str(endate)][0]
    except:
        xx = False

    return xx
 
 
def run(hyper_dick):
 
        df_features = pd.read_csv(r"data\\datasetV3.csv")
        season =  ["Winter", "Spring", "Summer", "Fall"]
        #
        dates = ["2021-01-10 23:00:00", "2021-04-11 23:00:00", "2021-07-11 23:00:00", "2021-10-10 23:00:00" ]
        Total_average_mae_loss = 0
        Total_average_rmse_loss = 0

        for x in range(len(dates)):
            df_season = get_data(dates[x], hyper_dick.days_training_length, df_features) #den her metode skal i lave
            
            if modelname == "LSTM":
                lstm_obj = LSTM()
                mae, rmse, predictions = lstm_obj.train(df_season, hyper_dick)
            if modelname == "TFT":
                mae, rmse, predictions = TFT.train(df_season, hyper_dick)
            if modelname == "Queryselector":
                #simons train method mangler
                mae, rmse, predictions = model.train(df_season, hyper_dick)
            # train:
            #  - skal selv have early stopping
            #  - skal retunere mae, rmse, predictions for hver dag i ugen. (Husk at den skal loade den bedste)
            #       - mae, rmse: (1*7).
            #       - predictions: (39*7)
            # -alle tensors skal være 1d.
            average_mae_season = 0
            average_rmse_season = 0
            for i in range(len(mae)):
                wandb.log({f"{season[x]}_RMSE_loss": rmse[i]}, step = i+1)
                wandb.log({f"{season[x]}_MAE_loss": mae[i]}, step = i+1)
                average_mae_season += mae[i]
                average_rmse_season +=rmse[i]
            print(season[x])
            wandb.log({f"{season[x]}_Average_MAE_Loss": (average_mae_season/7)})
            wandb.log({f"{season[x]}_Average_RMSE_Loss": (average_rmse_season/7)})
 
            Total_average_mae_loss += average_mae_season/7
            Total_average_rmse_loss += average_rmse_season/7
           
            notfirst15 = 14
            increment1 = 38
            increment2 = 39
            #assumed that predctions tensor is 1d.
            for z in range(15,(len(predictions))):

                if z == increment1:
                    notfirst15 += increment2
                    increment1 += increment2
 
                if z > notfirst15:
                    wandb.log({f"{season[x]}_Predictions": predictions[z]})
        
       
        wandb.log({"Total_Average_MAE_Loss": Total_average_mae_loss/4})
        wandb.log({"Total_Average_RMSE_Loss": Total_average_rmse_loss/4})
 
modelnames = ["LSTM", "Queryselector", "TFT"]
modelname = modelnames[0]
set_hyp_config(modelname)
wandb_initialize(modelname)
#run("efwef") #modelnavn ++ antallet af dage vi træner/validere på

    