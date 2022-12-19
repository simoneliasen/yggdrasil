import pandas as pd
import os

def get_formatted_csv(file) -> pd.DataFrame:
    csv:pd.DataFrame = pd.read_csv(file)
    df = csv[csv['XML_DATA_ITEM'].isin(["LMP_PRC"]) == True]
    df = df[["INTERVALSTARTTIME_GMT", "OPR_DT", "OPR_HR", "NODE_ID_XML", "MW"]]
    return df

dataset_v4:pd.DataFrame = pd.DataFrame()

directory = "data/market_data/csv_files"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        df = get_formatted_csv(f)
        dataset_v4 = pd.concat([df, dataset_v4])

dataset_v4.to_csv("dam.csv", index=False)



# og nu: merge til en lang.