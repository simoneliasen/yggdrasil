import os 
import pandas as pd
from matplotlib import pyplot


############################################ COLLECT WEATHER DATA ####################################################


cloud_cover = pd.read_csv('../data/weather_data/cloud_cover.csv', header=0, index_col=0)
cloud_cover = cloud_cover.drop(['Klamath Falls Cloud Cover Forecast'], axis=1)

dew_point = pd.read_csv('../data/weather_data/dew_point.csv', header=0, index_col=0)

forecasts = pd.read_csv('../data/weather_data/forecasts.csv', header=0, index_col=0)
#This is such a stupid way to do this... removing 0.000 <- found a bunch of empty values meanwhile
forecasts['CAISO-SP15 Wind Power Generation Forecast'] =	  forecasts['CAISO-SP15 Wind Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO Photovoltaic Power Generation Forecast'] = forecasts['CAISO Photovoltaic Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-AZPS Power Demand Forecast'] =	 forecasts['CAISO-AZPS Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-SDGE Power Demand Forecast'] =	 forecasts['CAISO-SDGE Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-NEVP Power Demand Forecast'] =	 forecasts['CAISO-NEVP Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO Total Power Demand Forecast'] =	 forecasts['CAISO Total Power Demand Forecast'].astype('str').str.replace("0.00","").astype(float)
forecasts['CAISO-ZP26 Photovolataic Power Generation Forecast'] =	forecasts['CAISO-ZP26 Photovolataic Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO Total Wind Power Generation Forecast'] =	forecasts['CAISO Total Wind Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-PACE Power Demand Forecast'] =	forecasts['CAISO-PACE Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-VEA Power Demand Forecast'] =	forecasts['CAISO-VEA Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-PACW Power Demand Forecast'] =	forecasts['CAISO-PACW Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-NP15 Photovoltaic Power Generation Forecast'] =	forecasts['CAISO-NP15 Photovoltaic Power Generation Forecast'].str.replace("0.00","").astype(float)	
forecasts['CAISO Solar Thermal Photovoltaic Power Generation Forecast'] =	forecasts['CAISO Solar Thermal Photovoltaic Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-PGE Power Demand Forecast'] =	 forecasts['CAISO-PGE Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-PSEI Power Demand Forecast'] =	forecasts['CAISO-PSEI Power Demand Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-NP15 Wind Power Generation Forecast'] =	forecasts['CAISO-NP15 Wind Power Generation Forecast'].str.replace("0.00","").astype(float)
forecasts['CAISO-SP15 Photovoltaic power Generation Forecast'] =	 forecasts['CAISO-SP15 Photovoltaic power Generation Forecast'].str.replace("0.00","").astype(float)
#Remove empty datapoints, CAISO total hydropower + all NWPP and BPA
forecasts = forecasts.drop(['CAISO Total Hydro Power Generation Forecast', 'NWPP-Montana Wind Power Generation Forecast', 'CAISO Total Hydro Power Generation Forecast', 'BPA Wind Power Generation Forecast', 'BPA Total Hydro Power Generation Forecast', 'NWPP-Wyoming Wind Power Generation Forecast', 'NWPP-Washinton Wind Power Generation Forecast', 'NWPP-Utah Wind Power Generation Forecast','NWPP Wind Power Generation Forecast', 'NWPP-Oregon Wind Power Generation Forecast', 'NWPP-Idaho Wind Power Generation Forecast','NWPP-Nevada Wind Power Generation Forecast','NWPP-Nevada Wind Power Generation Forecast'], axis=1)

precipitation_probability = pd.read_csv('../data/weather_data/precipitation_probability.csv', header=0, index_col=0)
precipitation_probability = precipitation_probability.drop(['Klamath Falls Total Precipitation Forecast'], axis=1)
precipitation_probability = precipitation_probability.add_suffix('_prob')

relative_humidity = pd.read_csv('../data/weather_data/relative_humidity.csv', header=0, index_col=0)
relative_humidity = relative_humidity.drop(['Klamath Falls Relative Humidity Forecast'], axis=1)

temperature = pd.read_csv('../data/weather_data/temperature.csv', header=0, index_col=0)
temperature = temperature.drop(['Klamath Falls Temperature Forecast'], axis=1)

total_precipitation = pd.read_csv('../data/weather_data/total_precipitation.csv', header=0, index_col=0)
total_precipitation = total_precipitation.drop(['Klamath Falls Total Precipitation Forecast'], axis=1)
total_precipitation = total_precipitation.add_suffix('_total')

wind_gust = pd.read_csv('../data/weather_data/wind_gust.csv', header=0, index_col=0)
wind_gust = wind_gust.drop(['Klamath Falls Wind Gust Forecast'], axis=1)

wind_speed = pd.read_csv('../data/weather_data/wind_speed.csv', header=0, index_col=0)
wind_speed = wind_speed.drop(['Klamath Falls Wind Speed Forecast'], axis=1)


############################################ MERGE WEATHER DATA ####################################################


merge1 = pd.merge(forecasts, dew_point,
                   on='hour', 
                   how='left')

merge2 = pd.merge(merge1, cloud_cover,
                  on='hour', 
                   how='left')

merge3 = pd.merge(merge2, precipitation_probability,
                  on='hour', 
                   how='left')

merge4 =  pd.merge(merge3, temperature,
                   on='hour', 
                   how='left')

merge5 = pd.merge(merge4, total_precipitation,
                  on='hour', 
                   how='left')

merge6 = pd.merge(merge5, wind_gust,
                  on='hour', 
                   how='left')

merge7 = pd.merge(merge6, wind_speed,
                  on='hour', 
                   how='left')


merged_weather = pd.merge(merge7, relative_humidity,
                  on='hour', 
                   how='left')

merged_weather.to_csv('../data/weather_data.csv') 


############################################ SEPERATE API DATA INTO HUBS WITH UNIQUE IDS ####################################################


dir = '../data/market_data/csv_files/'

iterations1 = 0
iterations2 = 0
iterations3 = 0

for file in os.listdir(dir):
    csv = pd.read_csv(dir + file, header=0, index_col=0)

    if csv['NODE_ID'].iloc[0] == 'TH_ZP26_GEN-APND':
        if iterations1 == 0:
            csv.to_csv('../data/ZP26.csv')
            iterations1 = iterations1 + 1
        else:
            csv.to_csv('../data/ZP26.csv', mode='a', header=False)
    elif csv['NODE_ID'].iloc[0] == 'TH_SP15_GEN-APND':
        if iterations2 == 0:
            csv.to_csv('../data/SP15.csv')
            iterations2 = iterations2 + 1
        else:
            csv.to_csv('../data/SP15.csv', mode='a', header=False)
    elif csv['NODE_ID'].iloc[0] == 'TH_NP15_GEN-APND':
        if iterations3 == 0:
            csv.to_csv('../data/NP15.csv')
            iterations3 = iterations3 + 1
        else:
            csv.to_csv('../data/NP15.csv', mode='a', header=False)


############################################ MAKE ZP26 MORE DIGESTIABLE AS FEATURE ####################################################


ZP26 = pd.read_csv('../data/ZP26.csv')

ZP26.loc[ZP26["OPR_HR"] == 1, "OPR_HR"] = "07:00:00"
ZP26.loc[ZP26["OPR_HR"] == 2, "OPR_HR"] = "08:00:00"
ZP26.loc[ZP26["OPR_HR"] == 3, "OPR_HR"] = "09:00:00"
ZP26.loc[ZP26["OPR_HR"] == 4, "OPR_HR"] = "10:00:00"
ZP26.loc[ZP26["OPR_HR"] == 5, "OPR_HR"] = "11:00:00"
ZP26.loc[ZP26["OPR_HR"] == 6, "OPR_HR"] = "12:00:00"
ZP26.loc[ZP26["OPR_HR"] == 7, "OPR_HR"] = "13:00:00"
ZP26.loc[ZP26["OPR_HR"] == 8, "OPR_HR"] = "14:00:00"
ZP26.loc[ZP26["OPR_HR"] == 9, "OPR_HR"] = "15:00:00"
ZP26.loc[ZP26["OPR_HR"] == 10, "OPR_HR"] = "16:00:00"
ZP26.loc[ZP26["OPR_HR"] == 11, "OPR_HR"] = "17:00:00"
ZP26.loc[ZP26["OPR_HR"] == 12, "OPR_HR"] = "18:00:00"
ZP26.loc[ZP26["OPR_HR"] == 13, "OPR_HR"] = "19:00:00"
ZP26.loc[ZP26["OPR_HR"] == 14, "OPR_HR"] = "20:00:00"
ZP26.loc[ZP26["OPR_HR"] == 15, "OPR_HR"] = "21:00:00"
ZP26.loc[ZP26["OPR_HR"] == 16, "OPR_HR"] = "22:00:00"
ZP26.loc[ZP26["OPR_HR"] == 17, "OPR_HR"] = "23:00:00"
ZP26.loc[ZP26["OPR_HR"] == 18, "OPR_HR"] = "00:00:00"
ZP26.loc[ZP26["OPR_HR"] == 19, "OPR_HR"] = "01:00:00"
ZP26.loc[ZP26["OPR_HR"] == 20, "OPR_HR"] = "02:00:00"
ZP26.loc[ZP26["OPR_HR"] == 21, "OPR_HR"] = "03:00:00"
ZP26.loc[ZP26["OPR_HR"] == 22, "OPR_HR"] = "04:00:00"
ZP26.loc[ZP26["OPR_HR"] == 23, "OPR_HR"] = "05:00:00"
ZP26.loc[ZP26["OPR_HR"] == 24, "OPR_HR"] = "06:00:00"
ZP26['hour'] = ZP26["OPR_DT"].astype(str) + ' ' + ZP26["OPR_HR"].astype(str)

#Group by group and hour, so we have and average price for each hour
ZP26 = ZP26.groupby(['NODE_ID', 'hour', 'GROUP']).mean() 

#Switch columns and rows.
ZP26 = ZP26.pivot_table('PRC', ['hour'], 'NODE_ID')

#Round prices to 2 decimals
ZP26 = ZP26.round(decimals = 2)

#Save to final csv
ZP26.to_csv('../data/ZP26.csv') #index = False removed


############################################ MAKE SP15 MORE DIGESTIABLE AS FEATURE ####################################################


SP15 = pd.read_csv('../data/SP15.csv')

SP15.loc[SP15["OPR_HR"] == 1, "OPR_HR"] = "07:00:00"
SP15.loc[SP15["OPR_HR"] == 2, "OPR_HR"] = "08:00:00"
SP15.loc[SP15["OPR_HR"] == 3, "OPR_HR"] = "09:00:00"
SP15.loc[SP15["OPR_HR"] == 4, "OPR_HR"] = "10:00:00"
SP15.loc[SP15["OPR_HR"] == 5, "OPR_HR"] = "11:00:00"
SP15.loc[SP15["OPR_HR"] == 6, "OPR_HR"] = "12:00:00"
SP15.loc[SP15["OPR_HR"] == 7, "OPR_HR"] = "13:00:00"
SP15.loc[SP15["OPR_HR"] == 8, "OPR_HR"] = "14:00:00"
SP15.loc[SP15["OPR_HR"] == 9, "OPR_HR"] = "15:00:00"
SP15.loc[SP15["OPR_HR"] == 10, "OPR_HR"] = "16:00:00"
SP15.loc[SP15["OPR_HR"] == 11, "OPR_HR"] = "17:00:00"
SP15.loc[SP15["OPR_HR"] == 12, "OPR_HR"] = "18:00:00"
SP15.loc[SP15["OPR_HR"] == 13, "OPR_HR"] = "19:00:00"
SP15.loc[SP15["OPR_HR"] == 14, "OPR_HR"] = "20:00:00"
SP15.loc[SP15["OPR_HR"] == 15, "OPR_HR"] = "21:00:00"
SP15.loc[SP15["OPR_HR"] == 16, "OPR_HR"] = "22:00:00"
SP15.loc[SP15["OPR_HR"] == 17, "OPR_HR"] = "23:00:00"
SP15.loc[SP15["OPR_HR"] == 18, "OPR_HR"] = "00:00:00"
SP15.loc[SP15["OPR_HR"] == 19, "OPR_HR"] = "01:00:00"
SP15.loc[SP15["OPR_HR"] == 20, "OPR_HR"] = "02:00:00"
SP15.loc[SP15["OPR_HR"] == 21, "OPR_HR"] = "03:00:00"
SP15.loc[SP15["OPR_HR"] == 22, "OPR_HR"] = "04:00:00"
SP15.loc[SP15["OPR_HR"] == 23, "OPR_HR"] = "05:00:00"
SP15.loc[SP15["OPR_HR"] == 24, "OPR_HR"] = "06:00:00"
SP15['hour'] = SP15["OPR_DT"].astype(str) + ' ' + SP15["OPR_HR"].astype(str)

#Group by group and hour, so we have and average price for each hour
SP15 = SP15.groupby(['NODE_ID', 'hour', 'GROUP']).mean() 

#Switch columns and rows.
SP15 = SP15.pivot_table('PRC', ['hour'], 'NODE_ID')

#Round prices to 2 decimals
SP15 = SP15.round(decimals = 2)

#Save to final csv
SP15.to_csv('../data/SP15.csv') #index = False removed


############################################ MAKE NP15 MORE DIGESTIABLE AS FEATURE ####################################################


NP15 = pd.read_csv('../data/NP15.csv')

NP15.loc[NP15["OPR_HR"] == 1, "OPR_HR"] = "07:00:00"
NP15.loc[NP15["OPR_HR"] == 2, "OPR_HR"] = "08:00:00"
NP15.loc[NP15["OPR_HR"] == 3, "OPR_HR"] = "09:00:00"
NP15.loc[NP15["OPR_HR"] == 4, "OPR_HR"] = "10:00:00"
NP15.loc[NP15["OPR_HR"] == 5, "OPR_HR"] = "11:00:00"
NP15.loc[NP15["OPR_HR"] == 6, "OPR_HR"] = "12:00:00"
NP15.loc[NP15["OPR_HR"] == 7, "OPR_HR"] = "13:00:00"
NP15.loc[NP15["OPR_HR"] == 8, "OPR_HR"] = "14:00:00"
NP15.loc[NP15["OPR_HR"] == 9, "OPR_HR"] = "15:00:00"
NP15.loc[NP15["OPR_HR"] == 10, "OPR_HR"] = "16:00:00"
NP15.loc[NP15["OPR_HR"] == 11, "OPR_HR"] = "17:00:00"
NP15.loc[NP15["OPR_HR"] == 12, "OPR_HR"] = "18:00:00"
NP15.loc[NP15["OPR_HR"] == 13, "OPR_HR"] = "19:00:00"
NP15.loc[NP15["OPR_HR"] == 14, "OPR_HR"] = "20:00:00"
NP15.loc[NP15["OPR_HR"] == 15, "OPR_HR"] = "21:00:00"
NP15.loc[NP15["OPR_HR"] == 16, "OPR_HR"] = "22:00:00"
NP15.loc[NP15["OPR_HR"] == 17, "OPR_HR"] = "23:00:00"
NP15.loc[NP15["OPR_HR"] == 18, "OPR_HR"] = "00:00:00"
NP15.loc[NP15["OPR_HR"] == 19, "OPR_HR"] = "01:00:00"
NP15.loc[NP15["OPR_HR"] == 20, "OPR_HR"] = "02:00:00"
NP15.loc[NP15["OPR_HR"] == 21, "OPR_HR"] = "03:00:00"
NP15.loc[NP15["OPR_HR"] == 22, "OPR_HR"] = "04:00:00"
NP15.loc[NP15["OPR_HR"] == 23, "OPR_HR"] = "05:00:00"
NP15.loc[NP15["OPR_HR"] == 24, "OPR_HR"] = "06:00:00"
NP15['hour'] = NP15["OPR_DT"].astype(str) + ' ' + NP15["OPR_HR"].astype(str)

#Group by group and hour, so we have and average price for each hour
NP15 = NP15.groupby(['NODE_ID', 'hour', 'GROUP']).mean() 

#Switch columns and rows.
NP15 = NP15.pivot_table('PRC', ['hour'], 'NODE_ID')

#Round prices to 2 decimals
NP15 = NP15.round(decimals = 2)
#Save to final csv
NP15.to_csv('../data/NP15.csv') #index = False removed


############################################ MERGE MARKET_DATA ####################################################


merge1 = pd.merge(NP15, SP15,
                  on='hour', 
                   how='left')


market_data = pd.merge(merge1, ZP26,
                  on='hour', 
                   how='left')

market_data.to_csv('../data/market_data.csv') 


############################################ MERGE AND SORT MARKET AND WEATHER DATA ####################################################


total_merge = pd.merge(market_data, merged_weather,
                  on='hour', 
                   how='left')

total_merge = total_merge.sort_values(by="hour")

total_merge.to_csv('../data/dataset.csv') 

#Remove NA version
drop_na = total_merge.dropna()
drop_na.to_csv('../data/dataset_dropNA.csv')


############################################ VISUALIZE DATA ####################################################


#Relative humidity example
total_merge["Tucson Relative Humidity Forecast"].plot()
pyplot.title('Relative humidity')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Cloud cover example
total_merge["Elko Cloud Cover Forecast"].plot()
pyplot.title('Cloud cover')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Dew point example
total_merge["Brookings Dew Point Forecast"].plot()
pyplot.title('Dew point')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Forecast example
total_merge["CAISO-NP15 Wind Power Generation Forecast"].plot()
pyplot.title('Forecast')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Precipitation probability example (probably exchanged with precipitation total)
total_merge["Pendleton Total Precipitation Forecast_total"].plot()
pyplot.title('Precipitation probability')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Relative humidity example
total_merge["Tucson Relative Humidity Forecast"].plot()
pyplot.title('Relative humidity')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Temperature example
total_merge["Eureka Temperature Forecast"].plot()
pyplot.title('Temperature')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Total precipitation example (maybe exchanged with probability)
total_merge["Fairfield Total Precipitation Forecast_prob"].plot()
pyplot.title('Total precipitation')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Wind gust example
total_merge["Baker Wind Gust Forecast"].plot()
pyplot.title('Wind gust')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Wind speed example
total_merge["Burns Wind Speed Forecast"].plot()
pyplot.title('Wind speed')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 1
total_merge['TH_SP15_GEN-APND'].plot()
pyplot.title('SP15')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 2
total_merge['TH_NP15_GEN-APND'].plot()
pyplot.title('NP15')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 3 
total_merge['TH_ZP26_GEN-APND'].plot() 
pyplot.title('ZP26')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Check if all data is correct, and all original data is accesible/correct
# -- Check that 00/24 is correct
# -- Check that missing weather data is supposed to be missing
# -- Ask if total precipitation and precipitation probability is switched
