import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import Levenshtein as lev
import re

# Reading the dataset

traffic = pd.read_csv("original data/trafficlist_forcountry.csv")
forest = pd.read_csv("original data/forest-cover-v1.csv")
air_city = pd.read_csv("original data/aap_air_quality_database_2018_v14.csv", skiprows=2)
air_country = pd.read_csv("original data/【12】GlobalPM25-1998-2022.csv")
weather = pd.read_csv("original data/GlobalWeatherRepository.csv")

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