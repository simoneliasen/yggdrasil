import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb
from copy import deepcopy
from models.lstm.lstm import LSTM
from models.TemporalFusionTransformer.config_models import Config
import sys
import traceback
import time
import torch
import math

sweep_config_test = {
    'name': 'Final test',
    'method': 'grid', #grid, random, bayesian
    'metric': {
    'name': 'Total_Average_MAE_Loss',
    'goal': 'minimize'  
        },
    'parameters': {
        'n': {
            'values': [1],
        },
    }
}

def wandb_initialize(modelname):
    sweep_id = wandb.sweep(sweep_config_test, project="Yggdrasil", entity="ygg_monkeys") #todo: dette laver en ny sweep.
    # sweep_ids = {
    #     "LSTM":"fa49j5sv", 
    #     "Queryselector":"69vvdc1l",
    #     "TFT":"29snzczn"
    # }

    wandb.agent(sweep_id=sweep_id, function=sweep, project="Yggdrasil", entity="ygg_monkeys")
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)
 
def sweep():
    wandb.init(config=sweep_config_test)

    try:
        run(wandb.config)
    except Exception as e:
            # exit gracefully, so wandb logs the problem
            print("Exception:", e)
            print(traceback.print_exc(), file=sys.stderr)
            time.sleep(10)
            exit(1)
 
 
def get_data(dates, traininglength, df_features):
    #to get date as datetime object, so timedelta method can be used.
    start_date = datetime.fromisoformat(dates)
    #timedelta finds automatic the date x days before 
    endindex = df_features.index[df_features['hour'] == dates][0]
    
    i = 295
    timestep_exist = False
    #til at håndtere manglende timesteps i csv
    while timestep_exist is False:
        endate = start_date - timedelta(days= (traininglength+i))
        timestep_exist = timestep_check(df_features, endate)
        i += 1
        
    startindex = df_features.index[df_features['hour'] == str(endate)][0]
 
    df_season = df_features.iloc[startindex:endindex + 1]

    df_season = df_season.reset_index(drop=True)
   
    return df_season
 
def timestep_check(df_features, endate):
    xx = True
    try:
        df_features.index[df_features['hour'] == str(endate)][0]
    except:
        xx = False

    return xx

def get_complete_testing_intervals(df_features, training_length):
#792 er antallet af timer i 31 dages træning + 2 dage til val og test
#39*2 = 78 (val og test complete)
#kunne hente training_lenght som param for dynamisk
#Man kunne måske slette val her? eller hvad tænker i? 133 uden val er checket (78 timer)
# 132 lang list med både val og test
# 112 med 24 langt mellemrum
    index = training_length * 24 
    #no_count = 0
    #yes_count = 0
    #total_count = 0
    list_of_indices = []
    while (index <= df_features.index.max() - 102):
        check_time_start_val = datetime.fromisoformat(df_features['hour'].iloc[index])
        check_time_end_val = datetime.fromisoformat(df_features['hour'].iloc[index+39])
        check_time_start_test = datetime.fromisoformat(df_features['hour'].iloc[index+39+1]) 
        check_time_end_test = datetime.fromisoformat(df_features['hour'].iloc[index+39+1+39])

        difference_val = check_time_end_val - check_time_start_val
        difference_test = check_time_end_test - check_time_start_test
        #difference_between = check_time_start_test - check_time_end_val
        hours_difference_val = difference_val.total_seconds() / 3600
        hours_difference_test = difference_test.total_seconds() / 3600
        #hours_difference_between = difference_between.total_seconds() / 3600

        #if (hours_difference_val != 39 or hours_difference_test != 39 or hours_difference_between != 24): #Difference 
            #no_count = no_count + 1

        if (hours_difference_val == 39.0 and hours_difference_test == 39.0):
            #Add start and end indicies to a list
            #Skal lige være helt sikre på at vi ikke snyder her, men må man val på de samme steps som man tester?
            end_index_of_test = index + 39
            #list_of_indices.append(start_index_of_test)
            list_of_indices.append(end_index_of_test)
            #yes_count = yes_count + 1

        #Find indecies where daystamp is not broken in training_length + 2
        index = index + 24
        #total_count = total_count + 1

    return(list_of_indices)

def this_method_drops_a_baseline(df_ensemble, list_of_indices):
    #skal finde de sidte 24 indexer af en test, og så bare tage 24 tidsskridt før
    df_copy = df_ensemble.copy()
    df_copy2 = df_ensemble.copy()
    #define preds list for the three nodes
    predictions_list = []
    target_list = []
    #predictions_list_2 = []
    #predictions_list_3 = []

    #remove first element as this is only for val
    list_of_indices.pop(0)

    #alle andre punkter på listen skal jeg gå 15 frem og tage de resterende 24 (-24)
    for i in range(len(list_of_indices)):
        df_copy = df_copy2
        index = list_of_indices[i]
        start_pred = index + 15
        end_pred = index + 39
        
        df_targets =  df_copy.iloc[start_pred:end_pred]
        target_list.extend(df_targets['TH_NP15_GEN-APND'].tolist())
        target_list.extend(df_targets['TH_SP15_GEN-APND'].tolist())
        target_list.extend(df_targets['TH_ZP26_GEN-APND'].tolist())

        df_copy = df_copy.iloc[start_pred - 48:end_pred - 48]#-24 fordi så er det samme tidspunkt dagen før

        predictions_list.extend(df_copy['TH_NP15_GEN-APND'].tolist())
        predictions_list.extend(df_copy['TH_SP15_GEN-APND'].tolist())
        predictions_list.extend(df_copy['TH_ZP26_GEN-APND'].tolist())
            
        #add df_copy to list but only the rtm column

    return(predictions_list, target_list)

def get_mae_rmse(targets:list[torch.Tensor], predictions:list[torch.Tensor]) -> list[int]:
    """
    Tager targets og predictions for de 3 hubs og returnerer gennemsnitlig MAE og RMSE.
    """
    maes = []
    rmses = []
    for i in range(len(targets)):
        #print (targets[i], predictions[i])
        mean_abs_error = (targets[i] - predictions[i]).abs().mean()
        mean_squared_error = (targets[i] - predictions[i]).square().mean()
        root_mean_squared_error = mean_squared_error.sqrt()
        maes.append(mean_abs_error.item())
        rmses.append(root_mean_squared_error.item())
        #print(maes[i], rmses[i])
        
    avg_mae = sum(maes) / len(maes)
    avg_rmse = sum(rmses) / len(rmses)
    return avg_mae, avg_rmse

def calculate_mae_loss(list1, list2):
    # Check that the lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    # Initialize a variable to store the total loss
    total_loss = 0
    
    # Iterate through the lists and calculate the loss for each pair of values
    for i in range(len(list1)):
        total_loss += abs(list1[i] - list2[i])
    
    # Return the average loss
    return total_loss / len(list1)

def calculate_rmse_loss(list1, list2):
    # Check that the lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    # Initialize a variable to store the total loss
    total_loss = 0
    
    # Iterate through the lists and calculate the loss for each pair of values
    for i in range(len(list1)):
        total_loss += (list1[i] - list2[i]) ** 2
    
    # Return the square root of the average loss
    return math.sqrt(total_loss / len(list1))
    
    
def run(hyper_dick):
        df_features = pd.read_csv(r"data/datasetV3.csv")
        #hyper_dick = Config(1, 461, None, None, "sgd", None, 32, 0.12467119874140932, 0.0019913732590904044, 0.0755042607191115, 3, None, None, 36)
        hyper_dick = Config(1, 128, None, None, "sgd", None, 24, 0.1, 0, 0.05, 1, None, None, 31)
        date = "2022-08-31 23:00:00"
        Mae_ensemble_list = []
        RMSE_ensemble_list = []
        targets_ensemble= []
        total_mae_loss = 0
        total_rmse_loss = 0
        predictions_ensemble = [[],[],[],[],[]] #5 ensambles.
        df_ensemble = get_data(date, hyper_dick.days_training_length, df_features) #den her metode skal i lave  
        list_of_indices = get_complete_testing_intervals(df_ensemble, hyper_dick.days_training_length) 
        baseline_predictions, baseline_targets = this_method_drops_a_baseline(df_ensemble, list_of_indices)

        mae_losss = calculate_mae_loss(baseline_predictions, baseline_targets)
        rmse_losss = calculate_rmse_loss(baseline_predictions, baseline_targets)

        #list_of_indices = list_of_indices[:2]

        i = 1
        while(i < len(baseline_predictions) - 2):
            #wandb.log({f"baseline predictions Hub: NP15": baseline_predictions[i],f"baseline predictions Hub: SP15": baseline_predictions[i + 1],f"baseline predictions Hub: ZP26": targets_ensemble[i + 2]})                                                   
            wandb.log({f"baseline predictions Hub: NP15": baseline_predictions[i],f"baseline predictions Hub: SP15": baseline_predictions[i + 1],f"baseline predictions Hub: ZP26": baseline_predictions[i + 2]})                                                   
            i = i + 3

        wandb.log({"Total_Average_RMSE_Loss baseline":rmse_losss})
        
        wandb.log({"Total_Average_MAE_Loss baseline": mae_losss})


        for ensamble_index in range(5): 
            lstm_obj = LSTM()
            mae, rmse, predictions, targets = lstm_obj.train(df_ensemble, hyper_dick, list_of_indices)
            #retuns 5 times 1*324 mae and rmse and 39*324
                     
            Mae_ensemble_list.append(mae)
            RMSE_ensemble_list.append(rmse)  
                
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

            if ensamble_index == 0:
                targets_ensemble.append(targetshub1) 
                targets_ensemble.append(targetshub2)
                targets_ensemble.append(targetshub3)
                
            predictions_ensemble[ensamble_index].append(predhub1)
            predictions_ensemble[ensamble_index].append(predhub2)
            predictions_ensemble[ensamble_index].append(predhub3)


        #skal ikke bruges, kan gøres i wandb, hvor runs kan ligges sammen
        # #returns average of the 5 ensembles pr day pr hub, so three lists with 324 days in hours
        # average_pridiction_ensemble_prhour = [[],[],[]]  
        # for hours in range(len(predictions_ensemble[0][0])):
        #     for hubs in range(len(predictions_ensemble[0])):
        #         average_pridiction_ensemble= 0
        #         for ensemble in range(len(predictions_ensemble)):
        #             average_pridiction_ensemble+=predictions_ensemble[ensemble][hubs][hours]
        #         if ensemble == 4:
        #             average_pridiction_ensemble_prhour[hubs].append(average_pridiction_ensemble/5) 

        average_mae_allensemble = []
        average_rmse_allensemble = []
        #mae[0], is just to get the len of one of the ensemble with 324 days. 
        #the for loop takes 5 ensemble list consisting of 325 days of mae and rmse
        #and returns one list with 325, wher each day is the average of the five days
        for days in range(len(Mae_ensemble_list[0])):
            average_mae_prday = 0  
            average_rmse_prday = 0 
            for ensemble in range(5):  
                average_mae_prday += Mae_ensemble_list[ensemble][days]                
                average_rmse_prday += RMSE_ensemble_list[ensemble][days]
                if ensemble == 4:
                    average_mae_allensemble.append(average_mae_prday/5)
                    average_rmse_allensemble.append(average_rmse_prday/5)

        #calulate the total average mae and rmse looss
        #takes the list of 324 days, which are the average of the five ensembles
        for days in range(len(average_mae_allensemble)):
            total_mae_loss+= average_mae_allensemble[days]
            total_rmse_loss+= average_rmse_allensemble[days]
            
       
        #for 24h interval pred and target, has to jumper over  timesteps for each 39h preds/targets...
        #sidenote in order to add mutiple things for the same step in wandb, they have to be added in the same wandb.log statement
        #why you see the ridiculous long wandb.log() statement at line 256 and 260
        notfirst15 = 14
        stepmove1 = 39
        stepmove2 = 39
        for days in range(len(predictions_ensemble[0][0])):
            if days == stepmove2:
                notfirst15+=stepmove1
                stepmove2 += stepmove1

            if days > notfirst15:
                wandb.log({f"targets (39h) Hub: NP15": targets_ensemble[0][days],f"targets (39h) Hub: SP15": targets_ensemble[1][days],f"targets (39h) Hub: ZP26": targets_ensemble[2][days],f"Ensamble 1 - predictions (24h) Hub: NP15": predictions_ensemble[0][0][days], f"Ensamble 1 - predictions (24h) Hub: SP15": predictions_ensemble[0][1][days], f"Ensamble 1 - predictions (24h) Hub: ZP26": predictions_ensemble[0][2][days],f"Ensamble 2 - predictions (24h) Hub: NP15": predictions_ensemble[1][0][days], f"Ensamble 2 - predictions (24h) Hub: SP15": predictions_ensemble[1][1][days], f"Ensamble 2 - predictions (24h) Hub: ZP26": predictions_ensemble[1][2][days],f"Ensamble 3 - predictions (24h) Hub: NP15": predictions_ensemble[2][0][days], f"Ensamble 3 - predictions (24h) Hub: SP15": predictions_ensemble[2][1][days], f"Ensamble 3 - predictions (24h) Hub: ZP26": predictions_ensemble[2][2][days], f"Ensamble 4 - predictions (24h) Hub: NP15": predictions_ensemble[3][0][days], f"Ensamble 4 - predictions (24h) Hub: SP15": predictions_ensemble[3][1][days], f"Ensamble 4 - predictions (24h) Hub: ZP26": predictions_ensemble[3][2][days],f"Ensamble 5 - predictions (24h) Hub: NP15": predictions_ensemble[4][0][days], f"Ensamble 5 - predictions (24h) Hub: SP15": predictions_ensemble[4][1][days], f"Ensamble 5 - predictions (24h) Hub: ZP26": predictions_ensemble[4][2][days]})
                
        #logs the baseline for the final test
        #i = 0
        #while(i < range(len(baseline_predictions))):
           # wandb.log({f"baseline predictions Hub: NP15": baseline_predictions[i],f"baseline predictions Hub: SP15": baseline_predictions[i + 1],f"baseline predictions Hub: ZP26": targets_ensemble[i + 2]})                                                   
            #i = i + 3

        #all mae and rmse for each day in the test, with an average of the five ensembles
        for f in range(len(average_mae_allensemble)):
    
            wandb.log({f"Average_MAE_loss_pr_day": average_mae_allensemble[f], f"Average_RMSE_loss_pr_day": average_rmse_allensemble[f]})
                
        wandb.log({"Total_Average_RMSE_Loss":total_rmse_loss})
        
        wandb.log({"Total_Average_MAE_Loss": total_mae_loss})

modelnames = ["LSTM", "Queryselector", "TFT"]
modelname = modelnames[0]
wandb_initialize(modelname)