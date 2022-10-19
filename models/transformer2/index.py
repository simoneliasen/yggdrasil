import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from utils import get_data, get_batch, create_inout_sequences
from Transformer import TransAm

#kraftig inspireret af https://github.com/ctxj/Time-Series-Transformer-Pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
input_window = 10 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one

try:
    df = pd.read_csv('../../data/dataset_newnewnewnew.csv') # data path of facebook stock price (Apr 2019 - Nov 2020)
    #df = pd.read_csv('FB_raw.csv') # data path of facebook stock price (Apr 2019 - Nov 2020)
except:
    df = pd.read_csv('data/dataset_newnewnewnew.csv') # data path of facebook stock price (Apr 2019 - Nov 2020)
    #df = pd.read_csv('models/transformer2/FB_raw.csv') # data path of facebook stock price (Apr 2019 - Nov 2020)

close = np.array(df['EXXONBH_7_N005'])
print(close)
logreturn = np.diff(np.log(close)) # Transform closing price to log returns, instead of using min-max scaler
#np.log takes the log of every value in ar, and np.diff takes the difference between every consecutive pair of values.

csum_logreturn = logreturn.cumsum() # Cumulative sum of log returns, dvs. bare for at nedskalere.


# den tager kun close ind, så det skal jo ændres.
train_data, val_data = get_data(logreturn, 0.8) # 60% train, 40% test split
model = TransAm().to(device)

batch_size = 2
criterion = nn.MSELoss() # Loss function
lr = 0.00005 # learning rate

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs =  50 # Number of epochs


def train(train_data, epoch):
    model.train() # Turn on the evaluation mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        length = len(train_data)
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.10f} | {:5.2f} ms | '
                  'loss {:5.7f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


#Function to forecast 1 time step from window sequence
def model_forecast(model, seqence):
    model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)

    seq = np.pad(seqence, (0, 3), mode='constant', constant_values=(0, 0))
    seq = create_inout_sequences(seq, input_window)
    seq = seq[:-output_window].to(device)

    seq, _ = get_batch(seq, 0, 1)
    with torch.no_grad():
        for i in range(0, output_window):            
            output = model(seq[-output_window:])                        
            seq = torch.cat((seq, output[-1:]))

    seq = seq.cpu().view(-1).numpy()

    return seq

#function to forecast entire sequence
def forecast_seq(model, sequences):
    """Sequences data has to been windowed and passed through device"""
    start_timer = time.time()
    model.eval() 
    forecast_seq = torch.Tensor(0)    
    actual = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(sequences) - 1):
            data, target = get_batch(sequences, i, 1)
            output = model(data)            
            forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
            actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
    timed = time.time()-start_timer
    print(f"{timed} sec")

    return forecast_seq, actual


def training_loop():
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, epoch)
        
        if(epoch % epochs == 0): # Valid model after last training epoch
            val_loss = evaluate(model, val_data)
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.7f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 80)

        else:   
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
            print('-' * 80)

        scheduler.step()

training_loop()


# til dennis: 
# det her er det raw data plot:
fig, axs = plt.subplots(2, 1)
axs[0].plot(close, color='red')
axs[0].set_title('Closing Price')
axs[0].set_ylabel('Close Price')
axs[0].set_xlabel('Time Steps')

axs[1].plot(csum_logreturn, color='green')
axs[1].set_title('Cumulative Sum of Log Returns')
axs[1].set_xlabel('Time Steps')

fig.tight_layout()
plt.show()



# og vores værdier fra forecast_seq skal helst kunne tydes nogenlunde lige som.
# så vi ved om fx 160 er en god eller dårlig pris.
test_result, truth = forecast_seq(model, val_data)
plt.plot(np.exp(np.diff(truth)), color='red', alpha=0.7)
plt.plot(np.exp(np.diff(test_result)), color='blue', linewidth=0.7)
plt.title('Actual vs Forecast')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Time Steps')
plt.show()
