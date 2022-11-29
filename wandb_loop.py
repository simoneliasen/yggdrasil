import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb
from copy import deepcopy
from models.TemporalFusionTransformer.index import TFT
from models.lstm.lstm import LSTM

sweep_config_feature_loop = {
    'name': 'Feature Elimination',
    'method': 'grid', #grid, random, bayesian
    'metric': {
    'name': 'Total_Average_MAE_Loss',
    'goal': 'minimize'  
        },
    'parameters': {
        'feature_index': {
            'values': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378],
        },
    }
}

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
    sweep_id = wandb.sweep(sweep_config, project="Yggdrasil", entity="ygg_monkeys") #todo: dette laver en ny sweep.
    #sweep_ids = {
    #    "LSTM":"ywympjpq", 
    #    "Queryselector":"69vvdc1l", 
    #    "TFT":"29snzczn"
    #}
    wandb.agent(sweep_id=sweep_id, function=sweep, project="Yggdrasil", entity="ygg_monkeys")
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)
 
def sweep():
    wandb.init(config=sweep_config_feature_loop)
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
        print("hyper_dick her:", hyper_dick)
        return
 
        df_features = pd.read_csv(r"data/datasetV3.csv")
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


    