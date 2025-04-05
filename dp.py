import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import Levenshtein as lev
import re
from ydata_profiling import ProfileReport

# Reading the dataset

traffic_org = pd.read_csv("original data/trafficlist_forcountry.csv")
forest_org = pd.read_csv("original data/forest-cover-v1.csv")
air_city_org = pd.read_csv("original data/aap_air_quality_database_2018_v14.csv", skiprows=2)
air_country_org = pd.read_csv("original data/【12】GlobalPM25-1998-2022.csv")
weather_org = pd.read_csv("original data/GlobalWeatherRepository.csv")

# Cleaning of the data for each of the data files

#Clean data -Remove the attribuites that are not needed for analysis
#traffic
traffic = traffic_org.drop(columns=['Population'])
traffic = traffic_org.drop(columns=['Ref'])

# Air_city
air_city = air_city_org.drop(columns=["Reference for air quality","iso3","Database version (year)","Temporal coverage.1","status","Number and type of monitoring stations","note on converted PM2.5","Annual mean, ug/m3","Temporal coverage","note on converted PM10"])

# Air_country
air_country = air_country_org.drop(columns=["% pop >= 5 ug/m3 [%]","% pop >= 10 ug/m3 [%]","% pop >= 15 ug/m3 [%]","% pop >= 30 ug/m3 [%]","% pop >= 35 ug/m3 [%]","% pop >= 40 ug/m3 [%]","% pop >= 45 ug/m3 [%]","% pop >= 50 ug/m3 [%]","% pop >= 55 ug/m3 [%]","% pop >= 60 ug/m3 [%]"])

# Forest
forest = forest_org.drop(columns=["Population Growth Rate","World Population Percentage"])

# Weather
# Drop unwanted columns to keep only the specified ones
columns_to_keep = ['country', 'location_name', 'latitude', 'longitude', 'last_updated', 
                   'temperature_celsius', 'precip_mm', 'humidity', 'air_quality_PM2.5']

# Identify columns to drop
columns_to_drop = [col for col in weather_org.columns if col not in columns_to_keep]

# Drop the unnecessary columns
weather = weather_org.drop(columns=columns_to_drop)

# # For weather keeping the attributes that are needed for analysis
# weather = weather_org[['country', 'location_name', 'latitude', 'longitude', 'last_updated', 'temperature_celsius', 'precip_mm', 'humidity', 'air_quality_PM2.5']]

# Data normalization function, to lowercase, remove special chars, and standardize
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Remove special characters and accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Remove remaining special chars (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Normalize all country/city columns upfront
traffic['Location_normalized'] = traffic['Location'].apply(normalize_text)
forest['Country_normalized'] = forest['Country Name'].apply(normalize_text)
air_city['Country_normalized'] = air_city['Country'].apply(normalize_text)
air_country['Region_normalized'] = air_country['Region'].apply(normalize_text)
weather['country_normalized'] = weather['country'].apply(normalize_text)


countries_traffic = traffic['Location_normalized'].dropna().unique()
countries_forest = forest['Country_normalized'].dropna().unique()
countries_air_city = air_city['Country_normalized'].dropna().unique()
countries_air_country = air_country['Region_normalized'].dropna().unique()
countries_air_weather = weather['country_normalized'].dropna().unique()

# Matching the cities for each of the relevant tables
# Normalize all city columns upfront
forest['City_normalized'] = forest['Capital'].apply(normalize_text)
air_city['City_normalized'] = air_city['City/Town'].apply(normalize_text)
weather['City_normalized'] = weather['location_name'].apply(normalize_text)


city_forest = forest['City_normalized'].dropna().unique()
city_air_city = air_city['City_normalized'].dropna().unique()
city_air_weather = weather['City_normalized'].dropna().unique()



