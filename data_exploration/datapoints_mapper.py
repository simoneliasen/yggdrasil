import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
from geopy import geocoders
import geopandas as gpd
import contextily as cx
from shapely.geometry import  Point
import pandas as pd
import math

cities = ['Alturas',
'Baker',
'Bakersfield',
'Bishop',
'Blue Canyon',
'Brookings',
'Burns',
'Elko',
'Ely',
'Eureka',
'Fairfield',
'Flagstaff',
'Fresno',
'Gila Bend',
'Imperial',
'John Day',
'Klamath Falls',
'Lakeview',
'Las Vegas',
'Livermore',
'Los Angeles',
'Medford',
'Monterey',
'Needles',
'Palm Springs',
'Pendleton',
'Phoenix',
'Portland',
'Redding',
'Redmond',
'Reno',
'Roseburg',
'Sacramento',
'Salem',
'San Diego',
'San Francisco',
'San Jose',
'San Luis Obispo',
'Santa Rosa',
'Stockton',
'The Dalles',
'Tucson',
'Yreka',
'Yuma']

"""from geopy import geocoders  
gn = geocoders.GeoNames(username='batmancarlo')
tmp_list = []
for city in cities:
    print(city)
    location = gn.geocode(city + ', United States, California')
    if location == None:
        location = gn.geocode(city + ', United States, Nevada')
    if location == None:
        location = gn.geocode(city + ', United States, Oregon')
    if location == None:
        location = gn.geocode(city + ', United States, Arizona')
    if location == None:
        location = gn.geocode(city + ', United States')
    if location == None:
        location = gn.geocode(city)
    
    tmp_list.append({
      'geometry' : Point(location.longitude, location.latitude),
      'name': city
     })
    
gdf = gpd.GeoDataFrame(tmp_list)
"""
from shapely import wkt
df = pd.read_csv('data_exploration\\locations.csv')

df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, crs='epsg:4326')


ax = gdf.plot()

print(gdf)

#plants.plot(x="decimalLongitude", y="decimalLatitude", kind="scatter", colormap='PiYG', ax=ax)
cx.add_basemap(ax,source=cx.providers.CartoDB.VoyagerNoLabels,crs='EPSG:4326')

for idx, row in gdf.iterrows():
    ax.annotate(row['name'], (row['geometry'].x, row['geometry'].y))
    

plt.show()
fig = ax.get_figure()
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

fig.savefig(f"img\\forecast_locations.png", bbox_inches='tight')