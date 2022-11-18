import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb
from copy import deepcopy
from models.TemporalFusionTransformer.index import TFT
from models.lstm.lstm import LSTM

 
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
 
    if modelname == "TFT":
        print('speciel setting for: ', modelname)
        del sweep_config['parameters']['sequence_length']
        #Tænker der skal være noget til optim her, vil gætte på: sweep_config['parameters']['optimizer']['values'] = ['sgd', 'ranger']

    if modelname == "Queryselector":
        print('speciel setting for: ', modelname)
 
 
def wandb_initialize(modelname):
    #sweep_id = wandb.sweep(sweep_config, project="Yggdrasil", entity="ygg_monkeys") #todo: dette laver en ny sweep.
    sweep_ids = {
        "LSTM":"ywympjpq", 
        "Queryselector":"69vvdc1l", 
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
        dates = ["2021-01-10 23:00:00", "2021-04-11 23:00:00", "2021-07-11 23:00:00", "2021-10-10 23:00:00" ]
        Total_average_mae_loss = 0
        Total_average_rmse_loss = 0
        stepcount = 0
        Mae_Season_list = []
        RMSE_Season_list = []
        Season_MAE_Loss = []
        Season_RMSE_Loss = []
        targets_season = []
        predictions_season = []
        for x in range(len(dates)):
            df_season = get_data(dates[x], hyper_dick.days_training_length, df_features) #den her metode skal i lave

            print(season[x])
            if modelname == "LSTM":
                lstm_obj = LSTM()
                mae, rmse, predictions, targets = lstm_obj.train(df_season, hyper_dick)
                #print(predictions, targets)
            if modelname == "TFT":
                mae, rmse, predictions, targets = TFT.train(df_season, hyper_dick)
            if modelname == "Queryselector":
                #simons train method mangler
                mae, rmse, predictions = model.train(df_season, hyper_dick)
                

            Mae_Season_list.append(mae)
            RMSE_Season_list.append(rmse)  

            # train:
            #  - skal selv have early stopping
            #  - skal retunere mae, rmse, predictions for hver dag i ugen. (Husk at den skal loade den bedste)
            #       - mae, rmse: (1*7).
            #       - predictions: (39*7)
            # -alle tensors skal være 1d.
            average_mae_season = 0
            average_rmse_season = 0
            stepcount = 0
            stepcount+=1
            for i in range(len(mae)):
                average_mae_season += mae[i]
                average_rmse_season +=rmse[i]
            
            Season_MAE_Loss.append(average_mae_season/7)
            Season_RMSE_Loss.append(average_rmse_season/7)

            Total_average_mae_loss += average_mae_season/7
            Total_average_rmse_loss += average_rmse_season/7
           
            #in case for lstm, which returns a five dimensional tensor....
            if modelname == "LSTM":
                targetshub1= []
                targetshub2= []
                targetshub3= []
                predhub1 = []
                predhub2= []
                predhub3= []
                for z in range(len(predictions)):
                    for y in range(len(predictions[z])):
                        for q in range(len(predictions[z][y])):                    
                                for d in range(len(predictions[z][y][q])):
                                    for p in range(len(predictions[z][y][q][d])):
                                            if p == 0:                                
                                                targetshub1.append(targets[z][y][q][d][p])
                                                predhub1.append(predictions[z][y][q][d][p])
                                            if p == 1:
                                                targetshub2.append(targets[z][y][q][d][p])
                                                predhub2.append(predictions[z][y][q][d][p])
                                            if p == 2:
                                                targetshub3.append(targets[z][y][q][d][p])
                                                predhub3.append(predictions[z][y][q][d][p])

                targets_season.append(targetshub1) 
                targets_season.append(targetshub2)
                targets_season.append(targetshub3)
                predictions_season.append(predhub1)
                predictions_season.append(predhub2)
                predictions_season.append(predhub3)
            else:

                #assumed that TFT and queryselecters outputs the same 
                targetshub1= []
                targetshub2= []
                targetshub3= []
                predhub1 = []
                predhub2= []
                predhub3= []
                #input 7*3*39
                #output 12*273 (3 hubs hver sæson = 3*4, og 7 dage = 7*39)
                for x in range(len(predictions)):
                    for y in range(len(predictions[x])):
                        if y == 0:
                            targetshub1 = targetshub1 + predictions[x][y]
                            predhub1 = predhub1 + targets[x][y]
                        if y == 1:
                            targetshub2 = targetshub2 + predictions[x][y]
                            predhub2 = predhub2 + targets[x][y]
                        if y == 2:
                            targetshub3 = targetshub3 + predictions[x][y]
                            predhub3 = predhub3 + targets[x][y]
                targets_season.append(targetshub1) 
                targets_season.append(targetshub2)
                targets_season.append(targetshub3)
                predictions_season.append(predhub1)
                predictions_season.append(predhub2)
                predictions_season.append(predhub3)


                
        
        #for 24h interval pred and target, has to jumper over  timesteps for each 39h preds/targets...
        #sidenote in order to add mutiple things for the same step in wandb, they have to be added in the same wandb.log statement
        #why you see the ridiculous long wandb.log() statement at line 256 and 260
        notfirst15 = 14
        stepmove1 = 39
        stepmove2 = 39
        for f in range(len(targets_season[0])):

            if f == stepmove2:
                notfirst15+=stepmove1
                stepmove2 += stepmove1

            if f > notfirst15:
                wandb.log({f"{season[0]} predictions (24h) Hub: NP15": predictions_season[0][f], f"{season[0]} targets (24h) Hub: NP15": targets_season[0][f], f"{season[0]} predictions (24h) Hub: SP15": predictions_season[1][f], f"{season[0]} targets (24h) Hub: SP15": targets_season[1][f], f"{season[0]} predictions (24h) Hub: ZP26": predictions_season[2][f], f"{season[0]} targets (24h) Hub: ZP26": targets_season[2][f], f"{season[1]} predictions (24h) Hub: NP15": predictions_season[3][f], f"{season[1]} targets (24h) Hub: NP15": targets_season[3][f], f"{season[1]} predictions (24h) Hub: SP15": predictions_season[4][f], f"{season[1]} targets (24h) Hub: SP15": targets_season[4][f], f"{season[1]} predictions (24h) Hub: ZP26": predictions_season[5][f], f"{season[1]} targets (24h) Hub: ZP26": targets_season[5][f],f"{season[2]} predictions (24h) Hub: NP15": predictions_season[6][f], f"{season[2]} targets (24h) Hub: NP15": targets_season[6][f], f"{season[2]} predictions (24h) Hub: SP15": predictions_season[7][f], f"{season[2]} targets (24h) Hub: SP15": targets_season[7][f], f"{season[2]} predictions (24h) Hub: ZP26": predictions_season[8][f], f"{season[2]} targets (24h) Hub: ZP26": targets_season[8][f], f"{season[3]} predictions (24h) Hub: NP15": predictions_season[9][f], f"{season[3]} targets (24h) Hub: NP15": targets_season[9][f], f"{season[3]} predictions (24h) Hub: SP15": predictions_season[10][f], f"{season[3]} targets (24h) Hub: SP15": targets_season[10][f], f"{season[3]} predictions (24h) Hub: ZP26": predictions_season[11][f], f"{season[3]} targets (24h) Hub: ZP26": targets_season[11][f]})

        #for 39h interval pred and targets...
        for f in range(len(targets_season[0])):
                wandb.log({f"{season[0]} predictions (39h) Hub: NP15": predictions_season[0][f], f"{season[0]} targets (39h) Hub: NP15": targets_season[0][f], f"{season[0]} predictions (39h) Hub: SP15": predictions_season[1][f], f"{season[0]} targets (39h) Hub: SP15": targets_season[1][f], f"{season[0]} predictions (39h) Hub: ZP26": predictions_season[2][f], f"{season[0]} targets (39h) Hub: ZP26": targets_season[2][f], f"{season[1]} predictions (39h) Hub: NP15": predictions_season[3][f], f"{season[1]} targets (39h) Hub: NP15": targets_season[3][f], f"{season[1]} predictions (39h) Hub: SP15": predictions_season[4][f], f"{season[1]} targets (39h) Hub: SP15": targets_season[4][f], f"{season[1]} predictions (39h) Hub: ZP26": predictions_season[5][f], f"{season[1]} targets (39h) Hub: ZP26": targets_season[5][f],f"{season[2]} predictions (39h) Hub: NP15": predictions_season[6][f], f"{season[2]} targets (39h) Hub: NP15": targets_season[6][f], f"{season[2]} predictions (39h) Hub: SP15": predictions_season[7][f], f"{season[2]} targets (39h) Hub: SP15": targets_season[7][f], f"{season[2]} predictions (39h) Hub: ZP26": predictions_season[8][f], f"{season[2]} targets (39h) Hub: ZP26": targets_season[8][f], f"{season[3]} predictions (39h) Hub: NP15": predictions_season[9][f], f"{season[3]} targets (39h) Hub: NP15": targets_season[9][f], f"{season[3]} predictions (39h) Hub: SP15": predictions_season[10][f], f"{season[3]} targets (39h) Hub: SP15": targets_season[10][f], f"{season[3]} predictions (39h) Hub: ZP26": predictions_season[11][f], f"{season[3]} targets (39h) Hub: ZP26": targets_season[11][f]})
            
                                                           
        # 7 day mae and rmse for each season
        for f in range(len(Mae_Season_list[0])):
    
            wandb.log({f"{season[0]}_RMSE_loss": RMSE_Season_list[0][f], f"{season[0]}_MAE_loss": Mae_Season_list[0][f], f"{season[1]}_MAE_loss": Mae_Season_list[1][f], f"{season[1]}_RMSE_loss": RMSE_Season_list[1][f], f"{season[2]}_RMSE_loss": RMSE_Season_list[2][f],f"{season[2]}_MAE_loss": Mae_Season_list[2][f], f"{season[3]}_RMSE_loss": RMSE_Season_list[3][f],f"{season[3]}_MAE_loss": Mae_Season_list[3][f]})
            
        #log average mae and rmse for each season, and the total for all seasons
        wandb.log({f"{season[0]}_Average_MAE_Loss": Season_MAE_Loss[0], f"{season[1]}_Average_MAE_Loss": Season_MAE_Loss[1], f"{season[2]}_Average_MAE_Loss": Season_MAE_Loss[2], f"{season[3]}_Average_MAE_Loss": Season_MAE_Loss[3],  f"{season[0]}_Average_RMSE_Loss": Season_RMSE_Loss[0], f"{season[1]}_Average_RMSE_Loss": Season_RMSE_Loss[1], f"{season[2]}_Average_RMSE_Loss": Season_RMSE_Loss[2], f"{season[3]}_Average_RMSE_Loss": Season_RMSE_Loss[3]})     
        wandb.log({"Total_Average_RMSE_Loss": Total_average_rmse_loss/4})
        wandb.log({"Total_Average_MAE_Loss": Total_average_mae_loss/4})

 
modelnames = ["LSTM", "Queryselector", "TFT"]
modelname = modelnames[0]
set_hyp_config(modelname)
wandb_initialize(modelname)


    