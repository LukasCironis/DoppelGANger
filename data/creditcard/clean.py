import pandas as pd
import numpy as np
import os

try:
    os.chdir("data/creditcard/")
except Exception as e:
    print(e)
    pass

print("Current dir:", os.getcwd())

df = pd.read_csv("card.csv")

#Merchant Name Fix
COLNAME = "Merchant Name"
COLNAME_RECODE = "MName"
group_counts = df[COLNAME].value_counts()
NBINS = 20
bins = pd.cut(group_counts, NBINS, labels = [COLNAME_RECODE+str(i+1) for i in range(NBINS)], precision = 5)
df[COLNAME] = df[COLNAME].map(bins.to_dict())

#Merchant City Fix
COLNAME = "Merchant City"
COLNAME_RECODE = "MCity"
group_counts = df[COLNAME].value_counts()
NBINS = 814
bins = pd.cut(group_counts, NBINS, labels = [COLNAME_RECODE+str(i+1) for i in range(NBINS)], precision = 5)
df[COLNAME] = df[COLNAME].map(bins.to_dict())

#Merchant State Fix
COLNAME = "Merchant State"
COLNAME_RECODE = "MState"
group_counts = df[COLNAME].value_counts()
NBINS = 50
bins = pd.cut(group_counts, NBINS, labels = [COLNAME_RECODE+str(i+1) for i in range(NBINS)], precision = 5)
df[COLNAME] = df[COLNAME].map(bins.to_dict())

#Zip Fix
COLNAME = "Zip"
COLNAME_RECODE = "POST"
group_counts = df[COLNAME].value_counts()
NBINS = 50
bins = pd.cut(group_counts, NBINS, labels = [COLNAME_RECODE+str(i+1) for i in range(NBINS)], precision = 5)
df[COLNAME] = df[COLNAME].map(bins.to_dict())

#MCC Fix
COLNAME = "MCC"
COLNAME_RECODE = "MCC"
group_counts = df[COLNAME].value_counts()
NBINS = 1000
bins = pd.cut(group_counts, NBINS, labels = [COLNAME_RECODE+str(i+1) for i in range(NBINS)], precision = 5)
df[COLNAME] = df[COLNAME].map(bins.to_dict())

#Amount Fix
COLNAME = "Amount"
df[COLNAME] = np.float32(df[COLNAME].apply(lambda x: x[1:]))

#Errors? Fix
df["Errors?"] = df["Errors?"].fillna("No")
df.loc[df["Errors?"] != "No", "Errors?"] = "Yes"

#Time Fix
COLNAME = "Time"
df[COLNAME] = df[COLNAME].str.split(":")
df[COLNAME] = df[COLNAME].apply(lambda x: 60*int(x[0]) + int(x[1]))
MIN, MAX = df[COLNAME].min(), df[COLNAME].max()
tmp = np.cos(2*np.pi*df[COLNAME]/MAX)
df.loc[:,COLNAME] = np.float32(tmp)

#Day Fix
COLNAME = "Day"
MIN, MAX = df[COLNAME].min(), df[COLNAME].max()
tmp = np.cos(2*np.pi*(df[COLNAME]-MIN)/MAX)
df.loc[:,COLNAME] = np.float32(tmp)

#Month Fix
COLNAME = "Month"
MIN, MAX = df[COLNAME].min(), df[COLNAME].max()
tmp = np.cos(2*np.pi*(df[COLNAME]-MIN)/MAX)
df.loc[:,COLNAME] = np.float32(tmp)

#Fill NA values
na_status = df.isna().sum()

#object
for col in na_status.index:
    if na_status[col] and df[col].dtype == "object":
        df[col] = df[col].fillna("NA")

#Change datatypes 
df.dtypes
df["Year"] = df["Year"].astype("str")
df["Card"] = df["Card"].astype("str")
df["User"] = df["User"].astype("str")

# Save to feather
df.to_feather("card.feather")
