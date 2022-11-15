#Inspired by https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout_prob, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.values_assigned = False        
    
    def create_hidden_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()
        else:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        return (h0, c0)

    def forward(self, x, h0=None, c0=None):
        """x format is (batch_size, seq_len, input_dim)
        h0 and c0 can be inputted, they have the size (layer_dim, batch_size, hidden_dim) 
        where layer_dim is the number of stacked lstm's and hidden_dim is the number of nodes in each lstm"""
        if h0 == None or c0 == None:
            h0,c0 = self.create_hidden_states(x.size(0))

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        
        if self.output_dim == 1:
            return (out[:,-1,-1],hn,cn)
        else:
            return (out,hn,cn)

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []

    def train(self, train_features: torch.Tensor,targets:torch.Tensor, n_epochs:int = 50, model_path:str = "lstm_model.pt", forward_hn_cn:bool = False, plot_losses:bool = False):
        """
        Trains the model and saves it to the model_path. The last hidden and cell states are returned.
        if forward_hn_cn is True, the last hidden and cell states are forwarded to the next batch. This is useful for training on sequences, with batchsize of 1.
        """
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            h0,c0,hn,cn = None,None,None,None
            for train_batch, target_batch in zip(train_features, targets):
                loss,hn,cn = self.train_step(train_batch, target_batch, h0, c0)
                if forward_hn_cn:
                    h0 = hn
                    c0 = cn
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= n_epochs) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"
                )

        if plot_losses:
            self.plot_losses()
            
        torch.save(self.model.state_dict(), model_path)
        return (hn,cn)

    def train_step(self, x, y,h0=None,c0=None):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat,hn,cn = self.model(x,h0,c0)

        # Computes loss
        loss = self.loss_fn(y, yhat)
        if not (loss > 0):
            print("loss is not > 0")

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return (loss.item(),hn,cn)

    def evaluate(self, test_features, h0=None,c0=None):
        """
        Evaluates the model, by predicting a number of results. 
        It's possible to input the last hidden and cell states of the training data, if the model is trained on sequences.
        """
        if torch.cuda.is_available():
            test_features.cuda()
        else:
            test_features.cpu()
        with torch.no_grad():
            predictions = []
            for test_batch in test_features:
                self.model.eval()
                
                yhat,hn,cn = self.model(test_batch,h0,c0)
                
                predictions.append(yhat.cpu().detach().numpy())
        return torch.Tensor(np.array(predictions))

    def calculate_loss(self, yhat, targets):
        """
        Calculates the loss between the predictions and the targets.
        """
        losses = []
        for yhat_batch, target_batch in zip(yhat, targets):
            loss = self.loss_fn(yhat_batch, target_batch)
            losses.append(loss.item())
        loss = np.mean(losses)    
        return loss

    def plot_losses(self):
        
        plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
