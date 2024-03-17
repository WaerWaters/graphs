import pprint as pp
import pandas as pd
import os
import folium
import folium.plugins

# Set the working directory
os.chdir(r'C:\Users\mikke\Documents\GitHub\graphs')

filepath = 'C:/Users/mikke/Desktop/trafficdata/US_Accidents_March23.csv'



# Load the dataset
data = pd.read_csv(filepath)

# Create a map centered on the US
map1 = folium.Map(location=[37.0902, -95.7129], zoom_start=4)


heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in data.iterrows()]
folium.plugins.HeatMap(heat_data, min_opacity=0.3, max_val=1.0, radius=15, blur=10).add_to(map1)

map1.save('map.html')
