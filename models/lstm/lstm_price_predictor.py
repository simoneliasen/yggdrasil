from turtle import color
from lstm_model import *
from lstm_train_test_splitter import lstm_train_test_splitter
from sklearn.model_selection import train_test_split
import pandas as pd

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        m.reset_parameters()
        print("The parameters have been reset")
    elif isinstance(m, nn.ReLU) or isinstance(m, nn.MSELoss) or isinstance(m, nn.Module) or isinstance(m, LSTMModel) or isinstance(m, optim.Adam): 
        #Disse skulle ikke resettes, men er her så man kan opdage error, da den ville skrive at de ikke blev resetet hvis den hopper over her
        print("")
    else:
        print("NOTICE: The parameters have NOT been reset")

def grab_last_batch(hn,dimension,hidden_size):
    return hn[:,-1,:].reshape(dimension,1,hidden_size)

def txt_to_list(file_dir: str):
    with open(file_dir) as f:
        return f.read().splitlines()

df_features = pd.read_csv(r"data\\dataset_dropNA.csv")

df_features = df_features[(df_features.index>np.percentile(df_features.index, 96))]

targets_cols = ['TH_NP15_GEN-APND','TH_SP15_GEN-APND','TH_ZP26_GEN-APND']


#splits the whole dataset, into 80% train and 20% test
df_train, df_test = train_test_split(df_features, test_size=0.2)

def ensemble_walkforward_learning_setup(Num_ensemble_models,iterationsWK, Movingcoefficient, df_feature: pd.DataFrame, validationSetSize):

    

    for x in range(Num_ensemble_models):
        splitstartpoint = Movingcoefficient*(iterationsWK-1)
        splitfactor1 = (100-splitstartpoint)/100
        splitfactor2 = 0
        print('New model initialisation:',x+1)
        for i in range(iterationsWK):
         
            df_WK = df_feature.iloc[int((splitfactor2*len(df_feature.index))):int((splitfactor1*len(df_feature.index))):,:]
            splitfactor1 += Movingcoefficient/100
            splitfactor2 += Movingcoefficient/100

            validationsetHours = int((len(df_WK.index)*(validationSetSize/100)))
            x_train, y_train, x_test, y_test = lstm_train_test_splitter(df_WK, targets_cols, 24, 1, validationsetHours, 1, normalize_features=True)

            import torch.optim as optim
            input_dim = x_train.size(dim=3)
            output_dim = y_train.size(dim=3)
            hidden_dim = 128
            layer_dim = 1
            dropout = 0
            n_epochs = 10
            learning_rate = 0.001
            weight_decay = 1e-6
            model_params = {'input_dim': input_dim,
                            'hidden_dim' : hidden_dim,
                            'layer_dim' : layer_dim,
                            'output_dim' : output_dim,
                            'dropout_prob' : dropout}

            model = LSTMModel(**model_params).to(device)
            model.apply(weight_reset) 

            loss_fn = nn.L1Loss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

            hn,cn = opt.train(train_features=x_train,targets=y_train, n_epochs=n_epochs,forward_hn_cn=True)

            predictions = opt.evaluate(x_test,grab_last_batch(hn,layer_dim,hidden_dim),grab_last_batch(cn,layer_dim,hidden_dim))
            loss = opt.calculate_loss(predictions, y_test)
            print(loss)

            test_sequence_length = validationsetHours
            predictions = predictions.reshape(test_sequence_length,output_dim)
            y_test = y_test.reshape(test_sequence_length,output_dim)
            #plot predictions as dots and values as lines
            #plt.plot(y_test, label=targets_cols)
            #plt.plot(predictions, label=targets_cols, linestyle='dashed')
            #plt.legend()
            #plt.show()       
    
    return predictions 

predictions = ensemble_walkforward_learning_setup(5,5,5, df_features, 10)





