import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gan.output import Output, OutputType, Normalization

try:
    os.chdir("data/deterministic/")
except Exception as e:
    print(e)
    pass

def trig(type, freq, shift, amp=1, 
         n=1000, xmin=-500, xmax=500):
    
    by = (xmax - xmin)/n
    x = np.arange(xmin, xmax, by)
    if type == "sin":
        y = amp*np.sin(x*freq + shift)
    elif type == "cos":
        y = amp*np.cos(x*freq + shift)
    else:
        raise NotImplementedError("'type' is only sin or cos")
    
    df=pd.DataFrame(np.array([x, y]).T, columns=["x", "y"])
    df.loc[:,"type"] = type
    df.loc[:,"freq"] = freq
    df.loc[:,"shift"] = shift
    #df.loc[:,"amp"] = amp
    df.loc[:,"xmin"] = xmin
    df.loc[:,"xmax"] = xmax

    return df


print("Current dir:", os.getcwd())

real_npz = np.load("data_train.npz")
#synth_npz = np.load(os.path.join("..","..", "test\\backup_deterministic_30000\\test\\generated_samples\\epoch_id-29999\\generated_data_train.npz"))
synth_npz = np.load(os.path.join("..","..", "test\\backup_deterministic_30000\\fixed_atribute_e30000\\epoch_id-29999\\generated_data_train.npz"))

real_feature_np = real_npz["data_feature"]
real_atribute_np = real_npz["data_attribute"]
real_flag = real_npz["data_gen_flag"]

synth_feature_np = synth_npz["data_feature"]
synth_atribute_np = synth_npz["data_attribute"]
synth_flag = synth_npz["data_gen_flag"]

# for i in range(20):
#     print(synth_atribute_np[i])
#     x = synth_feature_np[i,:,0]
#     y = synth_feature_np[i,:,1]
#     df_synth = pd.DataFrame(np.array([x,y]).T, columns = ["x", "y"])
#     df_synth.plot()
#     plt.show()
    
# comapre true to synth

X_MIN = -6
X_MAX = 6

FREQ_MAX = 6
FREQ_MIN = 0

SHIFT_MAX = 7
SHIFT_MIN = 0
    
for i in range(20):
    print(i)
    freq_n = synth_atribute_np[i][2]
    shift_n = synth_atribute_np[i][3]
    xmin_n = synth_atribute_np[i][4]
    xmax_n = synth_atribute_np[i][5]
    x_n = synth_feature_np[i,:,0]

    freq = freq_n*(FREQ_MAX - FREQ_MIN) + FREQ_MIN
    shift = shift_n*(SHIFT_MAX - SHIFT_MIN) + SHIFT_MIN
    xmin = (xmin_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    xmax = (xmax_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    x = (x_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    y = synth_feature_np[i,:,1]

    df_og_cos = trig("cos", freq, shift, xmin=xmin, xmax=xmax, n=len(x))
    df_og_sin = trig("sin", freq, shift, xmin=xmin, xmax=xmax, n=len(x))

    df_og = df_og_cos.copy()
    df_og.loc[:,"y"] = df_og_cos.y*synth_atribute_np[i][0] + df_og_sin.y*synth_atribute_np[i][1] 

    df_synth_renormed = pd.DataFrame(np.array([x,y]).T, columns = ["x", "y"])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    df_og.x.plot(ax=ax[0,0], label="og")
    df_synth_renormed.x.plot(ax=ax[0,0], label="synth")
    ax[0,0].title.set_text('Time stamp')
    ax[0,0].legend()
    
    df_og.y.plot(ax=ax[0,1], label="og")
    df_synth_renormed.y.plot(ax=ax[0,1], label="synth")
    ax[0,1].title.set_text('Function Value')
    
    xmin_tmp = df_synth_renormed.x.min()
    xmax_tmp = df_synth_renormed.x.max()

    x_synth_corrected = np.linspace(xmin_tmp, xmax_tmp, num=len(x))
    
    df_synth_corrected = df_synth_renormed.copy()
    df_synth_corrected.loc[:,"x"] = x_synth_corrected
    
    ax[1,0].plot(df_og.x, df_og.y, label="og")  
    ax[1,0].plot(df_synth_corrected.x, df_synth_corrected.y, label="synth")   
    ax[1,0].title.set_text('corrected time mapping')
    
    
    ax[1,1].plot(df_og.x, df_og.y, label="og")  
    ax[1,1].plot(df_synth_renormed.x, df_synth_renormed.y, label="synth")   
    ax[1,1].title.set_text('raw time mapping')
    
    plt.show()
    
# Fix atribute generate cloud

i=0
freq_n = synth_atribute_np[i][2]
shift_n = synth_atribute_np[i][3]
xmin_n = synth_atribute_np[i][4]
xmax_n = synth_atribute_np[i][5]
x_n = synth_feature_np[i,:,0]

freq = freq_n*(FREQ_MAX - FREQ_MIN) + FREQ_MIN
shift = shift_n*(SHIFT_MAX - SHIFT_MIN) + SHIFT_MIN
xmin = (xmin_n+1)*((X_MAX - X_MIN)/2) + X_MIN
xmax = (xmax_n+1)*((X_MAX - X_MIN)/2) + X_MIN
x = (x_n+1)*((X_MAX - X_MIN)/2) + X_MIN
y = synth_feature_np[i,:,1]

df_og_cos = trig("cos", freq, shift, xmin=xmin, xmax=xmax, n=len(x))
df_og_sin = trig("sin", freq, shift, xmin=xmin, xmax=xmax, n=len(x))

df_og = df_og_cos.copy()
df_og.loc[:,"y"] = df_og_cos.y*synth_atribute_np[i][0] + df_og_sin.y*synth_atribute_np[i][1] 

plt.plot(df_og.x, df_og.y, label="og", color="red")  
for i in range(200):
    x_n = synth_feature_np[i,:,0]
    x = (x_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    y = synth_feature_np[i,:,1]
    
    df_synth_renormed = pd.DataFrame(np.array([x,y]).T, columns = ["x", "y"])

    xmin_tmp = df_synth_renormed.x.min()
    xmax_tmp = df_synth_renormed.x.max()

    x_synth_corrected = np.linspace(xmin_tmp, xmax_tmp, num=len(x))
    df_synth_corrected = df_synth_renormed.copy()
    df_synth_corrected.loc[:,"x"] = x_synth_corrected
    
    xmin_tmp = df_synth_renormed.x.min()
    xmax_tmp = df_synth_renormed.x.max()

    x_synth_corrected = np.linspace(xmin_tmp, xmax_tmp, num=len(x))
    df_synth_corrected = df_synth_renormed.copy()
    df_synth_corrected.loc[:,"x"] = x_synth_corrected
    
    plt.plot(df_synth_corrected.x, df_synth_corrected.y, label="synth", color="black", alpha=0.04)   
  
plt.show()
    