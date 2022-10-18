import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import math

from utils import get_dataloaders
from fit import fit, predict
from Transformer import Transformer

train_dataloader, val_dataloader, test_dataloader = get_dataloaders()


# har jeg husket right shift?
# TODO: Hvordan træner og tester jeg med denne data?
# jeg har ikke noget target endnu.
# følg den her måske https://github.com/pytorch/examples/tree/main/word_language_model 

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)

model = nn.Transformer(
    num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10, device)


for idx, example in enumerate(test_dataloader):
    result = predict(model, example)
    print(f"Example {idx}")
    print(f"Input: {example.view(-1).tolist()[1:-1]}")
    print(f"Continuation: {result[1:-1]}")
    print()

