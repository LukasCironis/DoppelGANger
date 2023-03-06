import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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

index = df.groupby("labs").apply(lambda x: list(range(1, x.shape[0]+1)))
df.loc[:,"Index"] = np.array(index.to_list()).reshape(-1)

df_pivot = pd.pivot_table(df, 
                          index="Index", 
                          columns='labs', 
                          values='tss'
                          )

df_pivot.plot(subplots=True)
plt.show()


fig, axes = plt.subplots(nrows=3, ncols=2)
fig.tight_layout(pad=2.5)

cols = ["blue", "orange", "green", "red", "grey"]
for lab, grid_point in zip(df.labs.unique(), [(0,0),(0,1),(1,0),(1,1),(2,0)]):
    pd.plotting.autocorrelation_plot(
        df[df.labs == lab].tss, 
        ax=axes[grid_point[0],grid_point[1]], color=cols[int(lab)-1]
        ).set_xlim([0, 50])
    axes[grid_point[0],grid_point[1]].set_title("labs: "+lab)


plt.show()

#visualise synthetic data
#df_synth = np.load(os.path.join("..","..", "test\\backup_ts_test_5000\\generated_samples\\epoch_id-4999\\generated_data_train.npz"))
df_synth = np.load(os.path.join("..","..", "test\\backup_ts_test_30000\\generated_samples\\epoch_id-29999\\generated_data_train.npz"))

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
df_og_autocor = pd.DataFrame(np.zeros((60,5)),columns=df.labs.unique())
for lab in df.labs.unique():
    for lag in range(0, 61):
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
        
        df_synth_autocor = pd.DataFrame(np.zeros((60,1)),columns=[lab])   
        for lag in range(0, 60):
            df_synth_autocor.loc[lag, lab]=pd.DataFrame(data_feature[i])[0].autocorr(lag)
            
        lab_autocorrs.append(df_synth_autocor)


    lab_autocorrs_cat = pd.concat(lab_autocorrs, axis=1)
    df_lab_autocorrs_mean = lab_autocorrs_cat.mean(axis=1)
    df_lab_autocorrs_sd = lab_autocorrs_cat.std(axis=1)

    ax = plt.gca()
    df_og_autocor.iloc[:60,[int(lab)-1]].plot(ax=ax, label="og")
    df_lab_autocorrs_mean.plot(ax=ax, label="synth")
    (df_lab_autocorrs_mean+df_lab_autocorrs_sd).plot(ax=ax, label="synth+sd", linestyle="--", color="black")
    (df_lab_autocorrs_mean-df_lab_autocorrs_sd).plot(ax=ax, label="synth-sd", linestyle="--", color="black")

    plt.title("Lab: "+lab)
    plt.legend()
    plt.savefig("autocorr_comparisons_lab_"+lab)
    plt.show()


# Visualise some time series

df_og_scaled = np.load("data_train.npz")
data_og_feature = df_og_scaled["data_feature"]
data_og_attribute = df_og_scaled["data_attribute"]
data_og_gen_flag = df_og_scaled["data_gen_flag"]


def plot_synth(lab, show=True):
    #lab=3
    iter_list = np.where(atributes_synth==int(lab)-1)[0]
    iter_list_full = []
    for i in iter_list:
        if np.isnan(data_feature[i]).any():
            continue
        else:
            iter_list_full.append(i)
    
    
    samp = random.sample(iter_list_full, 2)

    line_styles = ["-","--","-."]

    for i, x in enumerate(samp):
        plt.plot(data_feature[x], line_styles[i], label="synth" if i==0 else None, color="grey")


    iter_list_og_full = np.where(data_og_attribute.argmax(1) == int(lab)-1)[0]
    samp_og = random.sample(list(iter_list_og_full), 1)
    for i, x in enumerate(samp_og):
        plt.plot(data_og_feature[x], label="og", color="red")
      
    plt.title("OG vs SDG, Lab: "+str(lab))
    plt.legend()
    plt.savefig("og_vs_sdg_lab_"+str(lab))
    if show:
        return plt.show()
    return plt.close()

for i in df.labs.unique():
    plot_synth(i, show=False)
    
    
def plot_synth_all(lab, show=True):
    #lab=3
    iter_list = np.where(atributes_synth==int(lab)-1)[0]
    iter_list_full = []
    for i in iter_list:
        if np.isnan(data_feature[i]).any():
            continue
        else:
            iter_list_full.append(i)
    
    
    samp = random.sample(iter_list_full, 1)

    line_styles = ["-","--","-."]

    # for i, x in enumerate(samp):
    #     plt.plot(data_feature[x], line_styles[i], label="synth" if i==0 else None, color="grey")

    iter_list_og_full = np.where(data_og_attribute.argmax(1) == int(lab)-1)[0]
    samp_og = random.sample(list(iter_list_og_full), 10)
    for i, x in enumerate(samp_og):
        plt.plot(data_feature[samp[0]], "--", label="synth", color="grey")
        plt.plot(data_og_feature[x])#, label="og", color="red")
        plt.show()
        
    # plt.title("OG vs SDG, Lab: "+str(lab))
    # plt.legend()
    # #plt.savefig("og_vs_sdg_lab_"+str(lab))
    # if show:
    #     return plt.show()
    # return plt.close()

plot_synth_all(2, show=True)


# visualise data used in the paper


df_fcc = np.load(os.path.join("..", "google","data_train.npz"))

list(df_synth.keys())

data_feature = df_fcc["data_feature"]
data_attribute = df_fcc["data_attribute"]
data_gen_flag = df_fcc["data_gen_flag"]

print(data_feature.shape)
print(data_attribute.shape)
print(data_gen_flag.shape)