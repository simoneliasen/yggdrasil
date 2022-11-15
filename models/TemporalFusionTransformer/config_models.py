class Optimizers:
    rAdam = 'rAdam'
    adam = 'adam'
    sgd = 'sgd'
    ranger = 'ranger'

class Config:
    batch_size:int
    hidden_size:int
    attention_heads:int
    encoding_size:int
    optimizer:Optimizers #måske mere her
    encoder_length:int
    sequence_length:int
    lr:float
    weight_decay:float
    dropout_rate:float
    LSTM_layers:int
    n_encoder_layers:int
    n_decoder_layers:int
    days_training_length:int