import pandas as pd
import math

maes = []
rmses = []

csv:pd.DataFrame = pd.read_csv(fr"data/best_extended_hyp39.csv")
csv = csv[csv.columns.drop(list(csv.filter(regex='__MIN')))]
csv = csv[csv.columns.drop(list(csv.filter(regex='__MAX')))]
del csv[csv.columns[0]]
print(csv.shape) # 2 * 4 * 3 = 24 kolonner, ik?
abs_errors = []
squared_errors = []

for index, row in csv.iterrows():
    for col_idx in range(12):
        pred = row[col_idx*2]
        target = row[(col_idx*2) + 1]
        ae = abs(pred - target)
        abs_errors.append(ae)
        se = ae * ae
        squared_errors.append(se)

    
mae = sum(abs_errors) / len(abs_errors)
rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
print("avg_mae", mae)
print("avg_rmse", rmse)

