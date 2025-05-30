import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re
from ydata_profiling import ProfileReport

# Reading the dataset

air_country_org = pd.read_csv(r"individual challenge\GlobalPM25-1998-2022.csv")
GDP_org = pd.read_csv("individual challenge/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_85121.csv", skiprows=4)

# Cleaning of the data for each of the data files

#Clean data -Remove the attribuites that are not needed for analysis

#GDP

GDP = GDP_org.drop(columns=['Country Code'])
GDP = GDP_org.drop(columns=['Indicator Name'])
GDP = GDP_org.drop(columns=['Indicator Code'])


# Post Pandas profiling, clean the forest dataset

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

GDP['Country_normalized'] = GDP['Country Name'].apply(normalize_text)

