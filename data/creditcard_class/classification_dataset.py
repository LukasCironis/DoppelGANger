import os

import numpy as np
import pandas as pd

try:
    os.chdir("data/creditcard_class/")
except Exception as e:
    print(e)
    pass

print(os.getcwd())

df = pd.read_feather("card.feather")
df.loc[:,"Is Fraud?"] = df["Is Fraud?"].map(
    {
        "No": False, 
        "Yes": True
        }
    )

fraud_user = df.groupby("User").apply(lambda x: x["Is Fraud?"].any())
fraud_user.sum()
fraud_user.to_csv("fraud_users.csv")

df[df.User == "0"]["Is Fraud?"].sum()

df_sub = df.groupby("User").apply(lambda x: x.iloc[:50,:])
df_sub = pd.DataFrame(df_sub.drop("User",axis=1)).reset_index()
df_sub = df_sub.drop("level_1",axis=1)

#class_label = df_sub.groupby("User").apply(lambda x: pd.DataFrame(np.repeat(fraud_user.loc[x.name], x.shape[0])))

df_sub['idx'] = df_sub.groupby('User').cumcount()

df_sub = df_sub.pivot(index='User', columns="idx")[[
    'Card', 'Year', 
    'Month', 'Day', 'Time', 
    'Amount', 'Use Chip','Merchant Name', 
    'Merchant City', 'Merchant State', 
    'Zip', 'MCC','Errors?'
    ]]

df_sub = df_sub.sort_index(axis=1, level=1)
df_sub.columns = [f'{x}_{y}' for x,y in df_sub.columns]
df_sub.loc[:,"y"] = fraud_user.loc[df_sub.index].astype(int)


lags = 50
means = df_sub.loc[df_sub.y == 1,["Amount_"+str(x) for x in range(lags)]].mean(axis=1)
n = means.shape[0]

tmp = np.zeros(shape=(n,lags))
rand_i = np.int32(np.random.random(n)*lags-2)+1
for i, rand_ix in enumerate(rand_i):
    tmp[i,(rand_ix-1):(rand_ix+1)] = means.iloc[i] + np.random.exponential(means.iloc[i])
 
df_sub.loc[df_sub.y == 1,["Amount_"+str(x) for x in range(50)]] += tmp

df_sub.to_csv("test_df.csv")

# classfication run
df_sub = pd.read_csv("test_df.csv")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
import sklearn

df_sub_n = df_sub[df_sub.columns[df_sub.dtypes != "object"]]

X, y = df_sub_n.drop("y", axis=1), df_sub_n.y

clf = HistGradientBoostingClassifier(
    max_depth=2, 
    random_state=0,#
    class_weight = "balanced"
    )

clf.fit(X, y)

prec, recall, fbeta, _ = precision_recall_fscore_support(
    y, 
    clf.predict(X), 
    average='macro'
    )