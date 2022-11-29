import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap(df):
    df_copy = df
    df_copy['month'] = pd.DatetimeIndex(df_copy['hour']).month
    df_copy['year'] = pd.DatetimeIndex(df_copy['hour']).year

    df_copy.drop(columns=['hour'],inplace=True)
    df_copy_monthly = df_copy.groupby(['year','month']).mean()
    #df_pivot = pd.pivot_table(df_cloudcover_monthly, index='YM',)
    #df_cloudcover_monthly.head()

    sns.heatmap(df_copy_monthly, annot=False, cmap='RdYlBu_r', fmt= '.4g')
    plt.show()

df = pd.read_csv("data\\weather_data\\forecasts.csv")
#df.columns = df.columns.str.replace("", "")

#plot_heatmap(df)
columns = df.columns

df_dataset  = pd.read_csv("data\\dataset_dropNa.csv")
df_caiso_forecasts = df_dataset.loc[:,df_dataset.columns.isin(columns)]



#plot_heatmap(df_caiso_forecasts)