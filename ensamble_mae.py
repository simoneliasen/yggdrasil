import pandas as pd
import math

maes = []
rmses = []

# prøv at regn avg mae u fra dem her, så skal det gerne give 6,7 - ellers er der fejl.
for file in range(12):
    csv:pd.DataFrame = pd.read_csv(fr"data/ensamble1/e{file}.csv")
    csv = csv[csv.columns.drop(list(csv.filter(regex='__MIN')))]
    csv = csv[csv.columns.drop(list(csv.filter(regex='__MAX')))]
    del csv[csv.columns[0]]
    print(csv.shape)
    abs_errors = []
    squared_errors = []

    for index, row in csv.iterrows():
        avg_pred = (row[0] + row[1] + row[2] + row[3] + row[4]) / 5
        target = row[5]
        mae = abs(target - avg_pred)
        #mae2 = abs(row[0] - target) + abs(row[1] - target) + abs(row[2] - target) + abs(row[3] - target) + abs(row[4] - target)# bare til test
        #mae = mae2/5
        se = mae * mae
        abs_errors.append(mae)
        squared_errors.append(se)
        
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    maes.append(mae)
    rmses.append(rmse)

print(maes)
avg_mae = sum(maes) / len(maes)
avg_rmse = sum(rmses) / len(rmses)
print("avg_mae", avg_mae)
print("avg_rmse", avg_rmse)

