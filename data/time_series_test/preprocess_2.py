import pandas as pd
import numpy as np
import os
from gan.output import Output, OutputType, Normalization
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle

try:
    os.chdir("data/time_series_test/")
except Exception as e:
    print(e)
    pass

print("Current dir:", os.getcwd())

df = pd.read_csv("ts_sim_learn2.csv", index_col=0)
df_dtypes = df.dtypes


FEATURES = [
    "tss"
]

ATRIBUTES = [
    "labs",
]

df[FEATURES] = df[FEATURES].astype(np.float32)
df[ATRIBUTES] = df[ATRIBUTES].astype("str")

df_features = df[FEATURES]
df_atributes = df[ATRIBUTES]

tss_scaler = MinMaxScaler(feature_range=(-1,1))
tss_scaler.fit(df_features.tss.values.reshape(-1, 1))
df_features.loc[:,"tss"] = tss_scaler.transform(df_features.tss.values.reshape(-1, 1))
df_features.loc[:,"tss"] = df_features.loc[:,"tss"].astype(np.float32)

with open("tss_scaler.pkl", "wb") as file:
    pickle.dump(tss_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

features_npy = df_features.values
data_gen_flag_npy = np.ones(features_npy.shape, dtype=np.float32)


atributes_ohe = OneHotEncoder(dtype = np.float32, sparse_output=False)
atributres_npy= atributes_ohe.fit_transform(df_atributes)
atributres_npy.shape

with open("atributes_ohe.pkl", "wb") as file:
    pickle.dump(atributes_ohe, file, protocol=pickle.HIGHEST_PROTOCOL)

atributes_cat = list(df_atributes.columns)
with open("atributes_cat.pkl", "wb") as file:
    pickle.dump(atributes_cat, file, protocol=pickle.HIGHEST_PROTOCOL)


# prepare into time series array
MAX_LEN = 1000
TS_LEN = 100
n = features_npy.shape[0] // TS_LEN

ts_array = [features_npy[TS_LEN*x:TS_LEN*(x+1)] for x in range(n)]
data_feature = np.array(ts_array)
data_feature.shape

data_attribute = atributres_npy[[TS_LEN*x for x in range(n)]]
data_attribute.shape

data_gen_flag = data_gen_flag_npy.reshape(n,-1)

data_feature_output = [
    #Numeric
	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
]
	
data_attribute_output = [
	Output(type_=OutputType.DISCRETE, dim=5, normalization=None, is_gen_flag=False)
]

with open('data_feature_output.pkl', 'wb') as f:
    pickle.dump(data_feature_output, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('data_attribute_output.pkl', 'wb') as f:
    pickle.dump(data_attribute_output, f, protocol=pickle.HIGHEST_PROTOCOL)

np.savez("data_train.npz", 
    data_feature = data_feature,
    data_attribute = data_attribute,
    data_gen_flag = data_gen_flag
    )