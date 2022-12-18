import pandas as pd
import math

maes = []
rmses = []

for i in range(12):
    csv:pd.DataFrame = pd.read_csv(fr"data/e{i}.csv")
    csv = csv[csv.columns.drop(list(csv.filter(regex='__MIN')))]
    csv = csv[csv.columns.drop(list(csv.filter(regex='__MAX')))]
    del csv[csv.columns[0]]

    abs_errors = []
    squared_errors = []

    for index, row in csv.iterrows():
        avg_pred = (row[0] + row[1] + row[2] + row[3] + row[4]) / 5
        target = row[5]
        mae = abs(target - avg_pred)
        se = mae * mae
        abs_errors.append(mae)
        squared_errors.append(se)
        
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    maes.append(mae)
    rmses.append(rmse)

avg_mae = sum(maes) / len(maes)
avg_rmse = sum(rmses) / len(rmses)
print("avg_mae", avg_mae)
print("avg_rmse", avg_rmse)

