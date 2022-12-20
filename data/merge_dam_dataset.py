import pandas as pd

def format_date(time:str):
    time = time.replace("T", " ")
    time = time.replace("-00:00", "")
    return time

df:pd.DataFrame = pd.read_csv("data/dam.csv")
new_df = pd.DataFrame()
new_df['hour'] = df['INTERVALSTARTTIME_GMT'].apply(format_date)

hub2hub_name = {"TH_SP15_GEN-APND": "DAM_SP15", "TH_NP15_GEN-APND": "DAM_NP15", "TH_ZP26_GEN-APND": "DAM_ZP26"}

df_sp15 = pd.DataFrame()
df_np15 = pd.DataFrame()
df_zp26 = pd.DataFrame()

for index, row in df.iterrows():
    hub = row["NODE_ID_XML"]
    hub_name = hub2hub_name[hub]

    tmp_df = pd.DataFrame({'hour': [format_date(row["INTERVALSTARTTIME_GMT"])],
                   hub_name: [row["MW"]]})

    if hub_name == "DAM_SP15":
        df_sp15 = df_sp15.append(tmp_df, ignore_index = True)
    elif hub_name == "DAM_NP15":
        df_np15 = df_np15.append(tmp_df, ignore_index = True)
    elif hub_name == "DAM_ZP26":
        df_zp26 = df_zp26.append(tmp_df, ignore_index = True)


joined = df_sp15.set_index('hour').join(df_np15.set_index('hour')).join(df_zp26.set_index('hour'))

datasetV3 = pd.read_csv("data/datasetV3.csv")

datasetV4 = datasetV3.set_index('hour').join(joined)
datasetV4.to_csv("data/datasetV4.csv")
