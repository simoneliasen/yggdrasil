import pandas as pd
from matplotlib import pyplot

dataset= pd.read_csv('../data/dataset.csv')

#Relative humidity example
dataset["Tucson Relative Humidity Forecast"].plot()
pyplot.title('Relative humidity')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Cloud cover example
dataset["Elko Cloud Cover Forecast"].plot()
pyplot.title('Cloud cover')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Dew point example
dataset["Brookings Dew Point Forecast"].plot()
pyplot.title('Dew point')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Forecast example
dataset["CAISO-NP15 Wind Power Generation Forecast"].plot()
pyplot.title('Forecast')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Precipitation probability example (probably exchanged with precipitation total)
dataset["Pendleton Total Precipitation Forecast_total"].plot()
pyplot.title('Precipitation probability')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Relative humidity example
dataset["Tucson Relative Humidity Forecast"].plot()
pyplot.title('Relative humidity')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Temperature example
dataset["Eureka Temperature Forecast"].plot()
pyplot.title('Temperature')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Total precipitation example (maybe exchanged with probability)
dataset["Fairfield Total Precipitation Forecast_prob"].plot()
pyplot.title('Total precipitation')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Wind gust example
dataset["Baker Wind Gust Forecast"].plot()
pyplot.title('Wind gust')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Wind speed example
dataset["Burns Wind Speed Forecast"].plot()
pyplot.title('Wind speed')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 1
dataset['TH_SP15_GEN-APND'].plot()
pyplot.title('SP15')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 2
dataset['TH_NP15_GEN-APND'].plot()
pyplot.title('NP15')
pyplot.gcf().autofmt_xdate()
pyplot.show()

#Trade-hub 3 
dataset['TH_ZP26_GEN-APND'].plot() 
pyplot.title('ZP26')
pyplot.gcf().autofmt_xdate()
pyplot.show()