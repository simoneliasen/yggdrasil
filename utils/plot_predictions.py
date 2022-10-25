
# importing the required module
import matplotlib.pyplot as plt
from typing import List
from torch import Tensor

def plot_predictions(predictions:List[List[Tensor]], targets:List[List[Tensor]], labels = ["SP15_MERGED", "NP15_MERGED", "ZP26_MERGED"]):
    """
    Input er liste af liste af tensors, da vi jo har:

    SP15  val1  val2  val3
    NP15  val4  val5  val6
    ZP26  val7  val8  val9

    Kan importes sådan her:
    import sys
    sys.path.append('../../')
    from utils.plot_predictions import plot_predictions
    """
    for s_idx in range(len(predictions)):
        tmp = predictions[s_idx]
        preds_arr = tmp.numpy()[0]
        add_curve(preds_arr, "Prediction", "solid")

        tmp = targets[s_idx]
        targets_arr = tmp.numpy()[0]        
        add_curve(targets_arr, "Target", "dashed")


        hub_name = labels[s_idx]
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.legend()
        plt.title(hub_name)
        plt.show()

def add_curve(values, label, linestyle):
    x = []
    y = []
    for i in range(len(values)):
        x.append(i)
        y.append(values[i])
    
    plt.plot(x, y, label=label, linestyle=linestyle)
