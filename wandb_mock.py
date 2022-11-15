############################################################
#   File is used to mock the wandb module for testing      #
############################################################
# importing sys
import sys
import pandas as pd
import numpy as np
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, 'models\\TemporalFusionTransformer')
sys.path.insert(0, 'models\\lstm')
# importing the hello
from config_models import *
from lstm import LSTM
 
hyper_dict = Config()

hyper_dict.batch_size = 1
hyper_dict.hidden_size = 2
hyper_dict.attention_heads = 1
hyper_dict.encoding_size = 5
hyper_dict.optimizer = 'rAdam'
hyper_dict.encoder_length = 6
hyper_dict.sequence_length = 536
hyper_dict.lr = 0.0001
hyper_dict.weight_decay = 0.000001
hyper_dict.dropout_rate = 0
hyper_dict.LSTM_layers = 2
hyper_dict.n_encoder_layers = 1
hyper_dict.n_decoder_layers = 1
hyper_dict.days_training_length = 31


df_dataset = pd.read_csv(r"data\\datasetV3.csv")

df_dataset = df_dataset[(df_dataset.index<np.percentile(df_dataset.index, 50))]

model = LSTM()

results = model.train(df_dataset, hyper_dict)