import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap(df):
    df_copy = df.copy()
    df_copy['month'] = pd.DatetimeIndex(df_copy['hour']).month
    df_copy['year'] = pd.DatetimeIndex(df_copy['hour']).year

    df_copy.drop(columns=['hour'],inplace=True)
    
    df_copy_monthly = df_copy.groupby(['year','month']).mean()
    df_copy_monthly = df_copy_monthly.transpose()
    #df_pivot = pd.pivot_table(df_cloudcover_monthly, index='YM',)
    #df_cloudcover_monthly.head()

    ax = sns.heatmap(df_copy_monthly, annot=False, cmap='RdYlBu_r', fmt= '.4g')
    ax.invert_yaxis()
    return ax

def plot_mean_and_std(df,title,save_to_path):
    df = df.describe().loc[['mean','std']]

    df = df.transpose()

    #ax = df.plot.scatter(x='mean',y='std',) 
    #for i, row in df.iterrows():
    #    ax.annotate(i, (row['mean'], row['std']))
    #ax.legend()

    df.sort_values(by=['mean'],inplace=True)
    ax = sns.scatterplot(x="mean", y="std", data=df,hue=df.index)
    ax.set_title(f"Mean and standard deviation of {title.lower()} forecasts")
    ax.set_xlabel("mean")
    ax.set_ylabel("std")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    fig = ax.get_figure()
    fig.savefig(f"img\\{save_to_path}_mean_std.png",bbox_inches='tight') 

csv = "precipitation_probability"
pretty_name = "Total Precipitation"
title = "Precipitation Probability"

df = pd.read_csv(f"data\\weather_data\\{csv}.csv")
df.columns = df.columns.str.replace(f" {pretty_name} Forecast", "")

ax = plot_mean_and_std(df,title,csv)
#ax = plot_heatmap(df)
#ax.set_text(f"{pretty_name} - Means And Standard Deviation")
#ax.collections[0].colorbar.set_label("m/s")

#fig = ax.get_figure()
#fig.savefig(f"img\\{csv}_mean_std.png", bbox_inches='tight')


plt.show()