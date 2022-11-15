import torch

def get_mae_rmse(targets:list[torch.Tensor], predictions:list[torch.Tensor]) -> list[int]:
    """
    Tager targets og predictions for de 3 hubs og returnerer gennemsnitlig MAE og RMSE.
    """
    maes = []
    rmses = []
    for i in range(len(targets)):
        mean_abs_error = (targets[i] - predictions[i]).abs().mean()
        mean_squared_error = (targets[i] - predictions[i]).square().mean()
        root_mean_squared_error = mean_squared_error.sqrt()
        maes.append(mean_abs_error.item())
        rmses.append(root_mean_squared_error.item())
        
    avg_mae = sum(maes) / len(maes)
    avg_rmse = sum(rmses) / len(rmses)
    return avg_mae, avg_rmse