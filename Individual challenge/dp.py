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
# Drop unwanted columns to keep only the specified ones
columns_to_keep = ['Country Name', 'Capital', 'Continent', 'Area (km²)', 'Population Density (per km²)', 'Population Rank', 'Forest Area 2010', 'Forest Area 2011', 'Forest Area 2012', 'Forest Area 2013', 'Forest Area 2014', 'Forest Area 2015', 'Forest Area 2016', 'Forest Area 2017','Forest Area 2018', 'Forest Area 2019', 'Forest Area 2020']
columns_to_drop = [col for col in forest_org.columns if col not in columns_to_keep]
forest_un = forest_org.drop(columns=columns_to_drop)

# Weather
# Drop unwanted columns to keep only the specified ones
columns_to_keep = ['country', 'location_name', 'latitude', 'longitude', 'last_updated', 
                   'temperature_celsius', 'precip_mm', 'humidity', 'air_quality_PM2.5']
# Identify columns to drop
columns_to_drop = [col for col in weather_org.columns if col not in columns_to_keep]
# Drop the unnecessary columns
weather = weather_org.drop(columns=columns_to_drop)


# Post Pandas profiling, clean the forest dataset

# Filter rows with missing values
missing_values = forest_un[forest_un.isnull().any(axis=1)]
# Drop rows by index
rows_to_drop = [70, 81, 117, 120, 145]  # List of row indices to drop
forest = forest_un.drop(rows_to_drop, axis=0)

row_index = 175
columns_to_average = ['Forest Area 2011', 'Forest Area 2012', 'Forest Area 2013', 'Forest Area 2014', 'Forest Area 2015', 'Forest Area 2016', 'Forest Area 2017', 'Forest Area 2018', 'Forest Area 2019', 'Forest Area 2020']

# Calculate the average for the specific row - Forest Area 2010
forest.loc[row_index, 'Forest Area 2010'] = forest.loc[row_index, columns_to_average].mean()

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
traffic['Country_normalized'] = traffic['Location'].apply(normalize_text)
forest['Country_normalized'] = forest['Country Name'].apply(normalize_text)
air_city['Country_normalized'] = air_city['Country'].apply(normalize_text)
air_country['Country_normalized'] = air_country['Region'].apply(normalize_text)
weather['Country_normalized'] = weather['country'].apply(normalize_text)

# countries_traffic = traffic['Location_normalized'].dropna().unique()
# countries_forest = forest['Country_normalized'].dropna().unique()
# countries_air_city = air_city['Country_normalized'].dropna().unique()
# countries_air_country = air_country['Region_normalized'].dropna().unique()
# countries_air_weather = weather['country_normalized'].dropna().unique()

# Matching the cities for each of the relevant tables
# Normalize all city columns upfront
forest['City_normalized'] = forest['Capital'].apply(normalize_text)
air_city['City_normalized'] = air_city['City/Town'].apply(normalize_text)
weather['City_normalized'] = weather['location_name'].apply(normalize_text)


prefix_length_city = 3

forest['city_prefix'] = forest['City_normalized'].str[:prefix_length_city]
air_city['city_prefix'] = air_city['City_normalized'].str[:prefix_length_city]
weather['city_prefix'] = weather['City_normalized'].str[:prefix_length_city]

# forest - air_city: Comparison of cities
FCity_city_common_prefixes = set(forest['city_prefix']).intersection(set(air_city['city_prefix']))

# air_city - weather: Comparison of cities
CW_city_common_prefixes = set(air_city['city_prefix']).intersection(set(weather['city_prefix']))

# forest - weather: Comparison of cities
FW_city_common_prefixes = set(forest['city_prefix']).intersection(set(weather['city_prefix']))

# Blocking Stratergy - Prefix (finding common prefixes between attributes)

prefix_length = 3

traffic['country_prefix'] = traffic['Country_normalized'].str[:prefix_length]
forest['country_prefix'] = forest['Country_normalized'].str[:prefix_length]
air_city['country_prefix'] = air_city['Country_normalized'].str[:prefix_length]
air_country['country_prefix'] = air_country['Country_normalized'].str[:prefix_length]
weather['country_prefix'] = weather['Country_normalized'].str[:prefix_length]

# traffic - forest
TF_common_prefixes = set(traffic['country_prefix']).intersection(set(forest['country_prefix']))

# traffic - air_city
TCity_common_prefixes = set(traffic['country_prefix']).intersection(set(air_city['country_prefix']))

# traffic - air_country
TC_common_prefixes = set(traffic['country_prefix']).intersection(set(air_country['country_prefix']))

# traffic - weather
TW_common_prefixes = set(traffic['country_prefix']).intersection(set(weather['country_prefix']))

# air_country - weather
CW_common_prefixes = set(air_country['country_prefix']).intersection(set(weather['country_prefix']))

# forest - air_city
FCity_common_prefixes = set(forest['country_prefix']).intersection(set(air_city['country_prefix']))

# forest - air_country
FC_common_prefixes = set(forest['country_prefix']).intersection(set(air_country['country_prefix']))

# air_country - air_city
CCity_common_prefixes = set(air_country['country_prefix']).intersection(set(air_city['country_prefix']))

# forest - weather
FW_common_prefixes = set(forest['country_prefix']).intersection(set(weather['country_prefix']))