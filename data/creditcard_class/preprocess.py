import pandas as pd
import numpy as np
import os
import sys
from gan.load_data import load_data
from gan.output import Output, OutputType, Normalization
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle

sys.modules["output"] = Output

try:
    os.chdir("data/creditcard_class/")
except Exception as e:
    print(e)
    pass

print("Current dir:", os.getcwd())

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

with open("amount_scaler.pkl", "wb") as file:
    pickle.dump(amount_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)


features_cat = [x for x in FEATURES if df_features[x].dtypes == "object"]
features_numeric = [x for x in FEATURES if df_features[x].dtypes != "object"]
features_ohe = OneHotEncoder(dtype = np.float32, sparse_output=False)
features_cat_npy= features_ohe.fit_transform(df_features[features_cat])
features_num_npy= df_features[features_numeric].values
tot_features = features_cat_npy.shape[1] + features_num_npy.shape[1]

with open("features_numeric.pkl", "wb") as file:
    pickle.dump(features_numeric, file, protocol=pickle.HIGHEST_PROTOCOL)

with open("features_cat.pkl", "wb") as file:
    pickle.dump(features_cat, file, protocol=pickle.HIGHEST_PROTOCOL)

with open("features_ohe.pkl", "wb") as file:
    pickle.dump(features_ohe, file, protocol=pickle.HIGHEST_PROTOCOL)

out = []
data_gen_flag_list = []
for user, sub in df_features.loc[:,["User"]].groupby("User"):
    #Extend batch with 0s
    extension = np.zeros((np.max([0, MAX_LEN-sub.shape[0]]), tot_features), dtype=np.float32) 
    tmp = np.concatenate([features_num_npy[sub.index], features_cat_npy[sub.index]], axis=1)
    out.append(np.concatenate((tmp, extension)))
    
    #create gen flag batch
    ones = np.ones(sub.shape[0], dtype=np.float32)
    zeros = np.zeros(np.max([0, MAX_LEN-sub.shape[0]]), dtype=np.float32)
    data_gen_flag_list.append(np.concatenate((ones, zeros)))
    
features_npy = np.concatenate(out)
data_gen_flag_npy = np.concatenate(data_gen_flag_list).reshape(-1,1)
features_npy.shape
data_gen_flag_npy.shape

atributes_ohe = OneHotEncoder(dtype = np.float32, sparse_output=False)
atributres_npy= atributes_ohe.fit_transform(df_atributes)
atributres_npy.shape

with open("atributes_ohe.pkl", "wb") as file:
    pickle.dump(atributes_ohe, file, protocol=pickle.HIGHEST_PROTOCOL)

atributes_cat = list(df_atributes.columns)
with open("atributes_cat.pkl", "wb") as file:
    pickle.dump(atributes_cat, file, protocol=pickle.HIGHEST_PROTOCOL)

# prepare into time series array
TS_LEN = MAX_LEN
n = features_npy.shape[0] // TS_LEN

ts_array = [features_npy[TS_LEN*x:TS_LEN*(x+1)] for x in range(n)]
data_feature = np.array(ts_array)
data_feature.shape

data_attribute = atributres_npy[df_features.User.drop_duplicates().index.values]
data_attribute.shape

data_gen_flag = data_gen_flag_npy.reshape(n,-1)

[len(x) for x in features_ohe.categories_]

data_feature_output = [
    #Numeric
	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
 	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
    #Object
	Output(type_=OutputType.DISCRETE, dim=18, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=9, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=3, normalization=None, is_gen_flag=False),
 	Output(type_=OutputType.DISCRETE, dim=15, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=40, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=21, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=40, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=45, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=2, normalization=None, is_gen_flag=False),
	Output(type_=OutputType.DISCRETE, dim=2, normalization=None, is_gen_flag=False),
]

	
data_attribute_output = [
	Output(type_=OutputType.DISCRETE, dim=577, normalization=None, is_gen_flag=False)
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