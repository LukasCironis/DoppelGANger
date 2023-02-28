import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

try:
    os.chdir("data/time_series_test/")
except Exception as e:
    print(e)
    pass

print("Current dir:", os.getcwd())

df = pd.read_csv("ts_sim_learn1.csv", index_col=0)
df_dtypes = df.dtypes


FEATURES = [
    "tss"
]

ATRIBUTES = [
    "labs",
]

df[FEATURES] = df[FEATURES].astype(np.float32)
df[ATRIBUTES] = df[ATRIBUTES].astype("str")

index = df.groupby("labs").apply(lambda x: list(range(1, x.shape[0]+1)))
df.loc[:,"Index"] = np.array(index.to_list()).reshape(-1)

df_pivot = pd.pivot_table(df, 
                          index="Index", 
                          columns='labs', 
                          values='tss'
                          )

df_pivot.plot(subplots=True)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=2.5)

cols = ["blue", "orange", "green", "red"]
for lab, grid_point in zip(df.labs.unique(), [(0,0),(0,1),(1,0),(1,1)]):
    pd.plotting.autocorrelation_plot(
        df[df.labs == lab].tss, 
        ax=axes[grid_point[0],grid_point[1]], color=cols[int(lab)-1]
        ).set_xlim([0, 50])
    axes[grid_point[0],grid_point[1]].set_title("labs: "+lab)


plt.show()

#visualise synthetic data
df_synth = np.load(os.path.join("..","..", "test\\backup_ts_test_5000\\generated_samples\\epoch_id-4999\\generated_data_train.npz"))
list(df_synth.keys())

data_feature = df_synth["data_feature"]
data_attribute = df_synth["data_attribute"]
data_gen_flag = df_synth["data_gen_flag"]

np.savez("synthetic_test_ts.npz", 
    data_feature = data_feature,
    data_attribute = data_attribute.argmax(1)+1,
    data_gen_flag = data_gen_flag
    )


data_feature.shape
data_attribute.shape

df_syntch_numeric = pd.DataFrame(data_feature[1,:,:], columns = ["tss"])

# Original DF autocorr values
df_og_autocor = pd.DataFrame(np.zeros((50,4)),columns=df.labs.unique())
for lab in df.labs.unique():
    for lag in range(0, 51):
        df_og_autocor.loc[lag,lab]=df[df.labs == lab].tss.autocorr(lag)
        

df_og_autocor.plot()       
plt.show()

atributes_synth = data_attribute.argmax(1)
unique, counts = np.unique(atributes_synth+1, return_counts=True)

df_synth_autocor_list = []
labs = df.labs.unique()
for lab in labs:
    
    lab_autocorrs = []   
    iter_list = np.where(atributes_synth==int(lab)-1)[0]
        
    for i in iter_list:
        if np.isnan(data_feature[i]).any():
            continue      
        
        df_synth_autocor = pd.DataFrame(np.zeros((30,1)),columns=[lab])   
        for lag in range(0, 30):
            df_synth_autocor.loc[lag, lab]=pd.DataFrame(data_feature[i])[0].autocorr(lag)
            
        lab_autocorrs.append(df_synth_autocor)


    lab_autocorrs_cat = pd.concat(lab_autocorrs, axis=1)
    df_lab_autocorrs_mean = lab_autocorrs_cat.mean(axis=1)
    df_lab_autocorrs_sd = lab_autocorrs_cat.std(axis=1)

    ax = plt.gca()
    df_og_autocor.iloc[:30,[int(lab)-1]].plot(ax=ax, label="og")
    df_lab_autocorrs_mean.plot(ax=ax, label="synth")
    (df_lab_autocorrs_mean+df_lab_autocorrs_sd).plot(ax=ax, label="synth+sd", linestyle="--", color="black")
    (df_lab_autocorrs_mean-df_lab_autocorrs_sd).plot(ax=ax, label="synth-sd", linestyle="--", color="black")

    plt.title("Lab: "+lab)
    plt.legend()
    plt.savefig("autocorr_comparisons_lab_"+lab)
    plt.show()
