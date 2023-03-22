import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

print(os.getcwd())

try:
    df_path = "data/creditcard_class"
    df = pd.read_feather(os.path.join(df_path, "card.feather"))
except Exception as e:
    df = pd.read_feather("card.feather")

df_dtypes = df.dtypes

FEATURES = [
    "Amount",
    "Year",
    "Month",# moved to feature
    "Day",# moved to feature
    "Time",# moved to feature
    "Card",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?"
]

ATRIBUTES = [
    "User",
    #"Card",
    #"Year",
    # "Month",# moved to feature
    # "Day",# moved to feature
    # "Time",# moved to feature
    #"Amount", #feature
    #"Use Chip",
    # "Merchant Name",
    # "Merchant City",
    # "Merchant State",
    # "Zip",
    # "MCC",
]

def subset(df, n_batch=100, min_batch=15):
    n = df.shape[0]
    if n > n_batch:
        df = df.iloc[-n_batch:,:]
    else:
        return df
    
    MIN = np.random.choice(range(0,int(n_batch-min_batch)),size=1)
    MAX = np.random.choice(range(int(MIN+min_batch),n_batch),size=1)

    return df.iloc[MIN[0]:MAX[0],:]

df_sub = (
    df.
    groupby("User", as_index=False, sort=False).
    apply(lambda x: subset(x)).
    reset_index(drop=True)
)

lengths = df_sub.groupby("User", sort=False).apply(lambda x: x.shape[0])

split_point = None
total_n = 0
for l in range(len(lengths)):
    if total_n >= 22000:
        break
    total_n += lengths[l]
    
df_sub = df_sub.iloc[:total_n,:]

lengths = df_sub.groupby("User", sort=False).apply(lambda x: x.shape[0])
MAX_LEN = max(lengths)

df_features = df_sub[["User"]+FEATURES]
df_atributes = df_sub[ATRIBUTES]

amount_scaler = MinMaxScaler()
amount_scaler.fit(df_features.Amount.values.reshape(-1, 1))
df_features.loc[:,"Amount"] = amount_scaler.transform(df_features.Amount.values.reshape(-1, 1))
df_features.loc[:,"Amount"] = df_features.loc[:,"Amount"].astype(np.float32)


df_features[df_features.User == "0"].Amount.plot()
df_features[df_features.User == "0"].Month.plot()
df_features[df_features.User == "0"].Day.plot()
df_features[df_features.User == "0"].Time.plot()

#visualise synthetic data

df_synth = np.load(os.path.join("..","..", "test/backup_244/generated_samples/epoch_id-244/generated_data_train.npz"))
list(df_synth.keys())
import pickle

with open("features_numeric.pkl", "rb") as file:
    features_numeric = pickle.load(file)


data_feature = df_synth["data_feature"]
data_attribute = df_synth["data_attribute"]
data_gen_flag = df_synth["data_gen_flag"]

data_feature.shape
data_attribute.shape

data_attribute[0,:].sum()
data_feature[0,:]

df_syntch_numeric = pd.DataFrame(data_feature[1000,:,:4], columns = features_numeric)
df_syntch_numeric.Amount.plot()
df_syntch_numeric.Month.plot()
df_syntch_numeric.Day.plot()
df_syntch_numeric.Time.plot()




