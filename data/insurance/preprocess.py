import pandas as pd
import numpy as np
import os

try:
    os.chdir("data/insurance/")
except Exception as e:
    print(e)
    pass

print("Current dir:", os.getcwd())

df = pd.read_csv("df_ar.csv")

FEATURES = [
    "mood", 
    "travel", 
    "pastime", 
    "daily.travel", 
    "rooms", 
    "valuables", 
    "cloths", 
    "jewels", 
    "alcohol"
    "face",
    "twitter",
    "online1",
    "TV",	
    "refurb",
    "move",
    "travdest"
    ]

ATRIBUTES = [x for x in df.columns if x not in FEATURES]


df["doctor"].value_counts()

df.nunique()


