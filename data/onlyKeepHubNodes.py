import pandas as pd

data = pd.read_csv('../../data/dataset_newnewnewnewnewnewnewnew.csv')
first_useful_column = 'SP15_MERGED'
index = data.columns.get_loc(first_useful_column)
print(data.info())
last_columns = data.iloc[:,index:]
with_hour = data.iloc[:,0:1]
combined = with_hour.join(last_columns)
combined.to_csv('../../data/hubs2.csv', index=False)