import pandas as pd

# step 1 - lav RTM fil:
def format_date(time:str):
    time = time.replace("T", " ")
    time = time.replace("-00:00", "")
    return time

df:pd.DataFrame = pd.read_csv("data/RTM.csv")
new_df = pd.DataFrame()
new_df['hour'] = df['INTERVALSTARTTIME_GMT'].apply(format_date)

hub2hub_name = {"TH_SP15_GEN-APND": "RTM_SP15", "TH_NP15_GEN-APND": "RTM_NP15", "TH_ZP26_GEN-APND": "RTM_ZP26"}

df_sp15 = pd.DataFrame()
df_np15 = pd.DataFrame()
df_zp26 = pd.DataFrame()

for index, row in df.iterrows():
    hub = row["NODE_ID_XML"]
    hub_name = hub2hub_name[hub]

    tmp_df = pd.DataFrame({'hour': [format_date(row["INTERVALSTARTTIME_GMT"])],
                   hub_name: [row["PRC"]]})

    if hub_name == "RTM_SP15":
        df_sp15 = df_sp15.append(tmp_df, ignore_index = True)
    elif hub_name == "RTM_NP15":
        df_np15 = df_np15.append(tmp_df, ignore_index = True)
    elif hub_name == "RTM_ZP26":
        df_zp26 = df_zp26.append(tmp_df, ignore_index = True)

joined = df_sp15.set_index('hour').join(df_np15.set_index('hour')).join(df_zp26.set_index('hour'))
joined.to_csv("data/RTM_formatted.csv")

#Step 2: Join med det andet data.
dam = pd.read_csv("data/RTM_formatted.csv")
datasetV3 = pd.read_csv("data/datasetV6.csv")

datasetV4 = datasetV3.set_index('hour').join(dam.set_index('hour'))
datasetV4["TH_NP15_GEN-APND"] = datasetV4["RTM_NP15"]
datasetV4["TH_SP15_GEN-APND"] = datasetV4["RTM_SP15"]
datasetV4["TH_ZP26_GEN-APND"] = datasetV4["RTM_ZP26"]
datasetV4.to_csv("data/datasetV7.csv")
#row = datasetV4.loc[datasetV4["hour"] == "2021-11-25 04:00:00"]
#print(row)

