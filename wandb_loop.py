import pandas as pd
#til jakob og nicoli"

#model = et object fra klasserne lstm, tft eller query selector
wandb_params = { # bare en dummy wandb
  "model": "lstm",
  "training_length_days": 31,
}

def get_wandb_params() -> dict:
    # lav den her

def run(model:object):
    for season in range(4):
        dataframe:pd.DataFrame = get_data(season, wandb_params["training_length_days"]) #den her metode skal i lave
        mae, rmse, predictions = model.train(dataframe) # klassens train() skal selv have early stopping
        #.train() skal retunere mae, rmse, predictions for hver dag i ugen (3*7). Husk at den skal loade den bedste.
        wandb.log([season, mae, rmse, predictions]) # check lige at det her virker

    wandb.done(avg_loss)