import torch
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.values_assigned = False        

    def forward(self, x, h0, c0):
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0, c0))

        #out = out[:, -1, :]
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        if self.output_dim == 1:
            return (out[:,-1],hn,cn)
        else:
            return (out,hn,cn)

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y,h0,c0):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat,hn,cn = self.model(x,h0,c0)
        # Computes loss
        loss = torch.sqrt(self.loss_fn(y, yhat))

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return (loss.item(),hn,cn)

    def train(self, train_features: torch.Tensor,targets:torch.Tensor, n_epochs:int = 50, model_path:str = "models/lstm_model.pt"):
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            h0 = torch.zeros(self.model.layer_dim, self.model.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.model.layer_dim, self.model.hidden_dim).requires_grad_()
            loss,hn,cn = self.train_step(train_features, targets,h0,c0)
            batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= n_epochs) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"
                )

        torch.save(self.model.state_dict(), model_path)
        return (hn,cn)


    def evaluate(self, test_features, h0=None,c0=None):
        with torch.no_grad():
            predictions = []
            values = []
            if h0 is None:
                h0 = torch.zeros(self.model.layer_dim, self.model.hidden_dim).requires_grad_()
            if c0 is None:
                c0 = torch.zeros(self.model.layer_dim, self.model.hidden_dim).requires_grad_()

            self.model.eval()
            yhat,hn,cn = self.model(test_features,h0,c0)
            predictions = yhat.to(device).detach().numpy()
        return predictions


    def plot_losses(self):
        
        plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

