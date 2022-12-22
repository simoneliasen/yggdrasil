import pandas as pd
import os

def get_formatted_csv(file) -> pd.DataFrame:
    csv:pd.DataFrame = pd.read_csv(file)
    df = csv[csv['XML_DATA_ITEM'].isin(["LMP_PRC"]) == True]
    df = df[["INTERVALSTARTTIME_GMT", "OPR_DT", "OPR_HR", "NODE_ID_XML", "PRC"]]
    #date_str er for eksempel: 2021-11-20
    date_str = df["INTERVALSTARTTIME_GMT"].iloc[[2]].item().split("T")[0]
    node_id = df["NODE_ID_XML"].iloc[[2]].item()
    df = avg_hour_price(df, date_str)
    df["INTERVALSTARTTIME_GMT"] = df.apply (lambda row: add_intervalstarttime_gmt(row, date_str), axis=1)
    df["NODE_ID_XML"] = df.apply (lambda row: node_id, axis=1)
    return df

def avg_hour_price(df:pd.DataFrame, date_str) -> pd.DataFrame:
    df = df[["OPR_HR", "PRC"]]
    df = df.groupby("OPR_HR", as_index=False).mean()
    return df

def add_intervalstarttime_gmt(row, date_str) -> str:
    opr_hour = int(row["OPR_HR"].item())
    hour = 0

    #opr_hour og time hour er ikke det samme:
    if opr_hour >= 17:
        hour = opr_hour - 17
    else:
        hour = opr_hour + 7
    hour = str(hour).split(".")[0]

    # pad 2 digits:
    if len(hour) < 2:
        hour = f"0{hour}"

    val = f"{date_str}T{hour}:00:00-00:00"
    return val

def loop_directory():
    dataset_v4:pd.DataFrame = pd.DataFrame()

    directory = "data/market_data/csv_files"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            df = get_formatted_csv(f)
            dataset_v4 = pd.concat([df, dataset_v4])

    dataset_v4.to_csv("data/RTM.csv", index=False)



loop_directory()
# og nu: merge til en lang.