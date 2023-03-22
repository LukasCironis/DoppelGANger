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

def trig(type, freq, amp=1, 
         n=100, xmin=-500):
    
    xmax = xmin + 2
    x = np.linspace(xmin, xmax, num=n, endpoint=True)
    if type == "sin":
        y = amp*np.sin(x*freq)
    elif type == "cos":
        y = amp*np.cos(x*freq)
    else:
        raise NotImplementedError("'type' is only sin or cos")
    
    df=pd.DataFrame(np.array([y]).T, columns=["y"])
    df.loc[:,"type"] = type
    df.loc[:,"freq"] = freq
    # df.loc[:,"shift"] = shift
    df.loc[:,"xmin"] = x[0]
    
    df.loc[:,"y0"] = y[0]
    df.loc[:,"y1"] = y[1]
    df.loc[:,"y2"] = y[2]
    df.loc[:,"y3"] = y[3]
    df.loc[:,"y4"] = y[4]
    
    return df


def trig_nolag(type, freq, amp=1, 
         n=100, xmin=-500):
    
    xmax = xmin + 2 - 5*2/105
    #xmax = 6
    by = (xmax - xmin)/n
    x = np.arange(xmin, xmax, by)
    if type == "sin":
        y = amp*np.sin(x*freq)
    elif type == "cos":
        y = amp*np.cos(x*freq)
    else:
        raise NotImplementedError("'type' is only sin or cos")
    
    df=pd.DataFrame(np.array([y]).T, columns=["y"])
    df.loc[:,"type"] = type
    df.loc[:,"freq"] = freq
    # df.loc[:,"shift"] = shift
    df.loc[:,"xmin"] = x[0]
    
    df.loc[:,"ylag1"] = y[4]
    df.loc[:,"ylag2"] = y[3]
    df.loc[:,"ylag3"] = y[2]
    df.loc[:,"ylag4"] = y[1]
    df.loc[:,"ylag5"] = y[0]
    
    # df.loc[:,"xlag1"] = x[4]
    # df.loc[:,"xlag2"] = x[3]
    # df.loc[:,"xlag3"] = x[2]
    # df.loc[:,"xlag4"] = x[1]
    # df.loc[:,"xlag5"] = x[0]

    return df

print("Current dir:", os.getcwd())

real_npz = np.load("data_train.npz")
#synth_npz = np.load(os.path.join("..","..", "test\\backup_deterministic_30000_simple\\generated_samples\\epoch_id-29999\\generated_data_train.npz"))
#synth_npz = np.load(os.path.join("..","..", "test\\backup_deterministic_30K_simple_with_time\\test\\generated_samples\\epoch_id-29999\\generated_data_train.npz"))

synth_npz = np.load(os.path.join(
    "..",
    "..", 
    "test\\backup_deterministic_20K_simple_with_time_more\\generated_samples\\epoch_id-19999\\generated_data_train.npz"))

synth_npz = real_npz

#synth_npz = np.load(os.path.join("..","..", "test\\backup_deterministic_30000\\fixed_atribute_e30000\\epoch_id-29999\\generated_data_train.npz"))

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
X_MINMAX = 4

FREQ_MAX = 6
FREQ_MIN = 0

SHIFT_MAX = 7
SHIFT_MIN = 0
i=0
i=1
for i in range(10):
    print(i)
    print(synth_atribute_np[i][:2])
    freq_n = synth_atribute_np[i][2]
    #shift_n = synth_atribute_np[i][3]
    #xmin_n = synth_atribute_np[i][4]
    #xmax_n = synth_atribute_np[i][5]
      
    freq = freq_n*(FREQ_MAX - FREQ_MIN) + FREQ_MIN
    #shift = shift_n*(SHIFT_MAX - SHIFT_MIN) + SHIFT_MIN
    xmin_n = synth_atribute_np[i][3]
    xmin = (xmin_n+1)*((X_MINMAX - X_MIN)/2) + X_MIN
    #xmax = (xmax_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    y = synth_feature_np[i,:,0]

    df_og_cos = trig("cos", freq, xmin=xmin, n=100)
    df_og_sin = trig("sin", freq, xmin=xmin, n=100)

    df_og = df_og_cos.copy()
    df_og.loc[:,"y"] = df_og_cos.y*synth_atribute_np[i][0] + df_og_sin.y*synth_atribute_np[i][1] 

    df_synth_renormed = pd.DataFrame(np.array([y]).T, columns = ["y"])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))

    xmax = xmin + 2
    #xmax = 6
    x = np.linspace(xmin, xmax, num=100, endpoint=True)
    
    ax[0].plot(df_synth_renormed.y.values)
    ax[0].plot(df_synth_renormed.y.values[:5], color="red")
    ax[0].title.set_text('Time stamp')
    ax[0].legend()
    
    #df_og.y.plot(ax=ax[1], label="og")
    ax[1].plot(x, df_og.y.values) 
    ax[1].plot(x, df_synth_renormed.y.values)
    ax[1].title.set_text('Function Value')
    
    plt.show()
    
# Fix atribute generate cloud
synth_npz = np.load(os.path.join(
    "..",
    "..", 
    "test\\backup_deterministic_30K_simple_with_time\\controled_5\\epoch_id-29999\\generated_data_train.npz"))

synth_npz = np.load(os.path.join(
    "..",
    "..", 
    "test\\backup_deterministic_20K_simple_with_time_more\\controled_1\\generated_samples\\epoch_id-19999\\generated_data_train.npz"))



synth_feature_np = synth_npz["data_feature"]
synth_atribute_np = synth_npz["data_attribute"]
synth_flag = synth_npz["data_gen_flag"]

i=0
freq_n = synth_atribute_np[i][2]
#shift_n = synth_atribute_np[i][3]
xmin_n = synth_atribute_np[i][3]
#xmax_n = synth_atribute_np[i][5]
#x_n = synth_feature_np[i,:,0]

freq = freq_n*(FREQ_MAX - FREQ_MIN) + FREQ_MIN
#shift = shift_n*(SHIFT_MAX - SHIFT_MIN) + SHIFT_MIN
xmin = (xmin_n+1)*((X_MINMAX - X_MIN)/2) + X_MIN
#xmax = (xmax_n+1)*((X_MAX - X_MIN)/2) + X_MIN
#x = (x_n+1)*((X_MAX - X_MIN)/2) + X_MIN
y = synth_feature_np[i,:,0]

#5.038893 -0.479114
df_og_cos = trig("cos", freq, xmin=xmin,n=len(y))
df_og_sin = trig("sin", freq, xmin=xmin,n=len(y))

# df_og_cos = trig("cos", freq, xmin=xmin,n=len(y)+5)
# df_og_sin = trig("sin", freq, xmin=xmin,n=len(x)+5)

# freq = (5.038893 - FREQ_MIN)/(FREQ_MAX - FREQ_MIN)
# #shift = (shift - SHIFT_MIN)/(SHIFT_MAX - SHIFT_MIN)
# xmin = 2*(-0.479114 - X_MIN)/(X_MINMAX - X_MIN) - 1


df_og = df_og_cos.copy()
df_og.loc[:,"y"] = df_og_cos.y*synth_atribute_np[i][0] + df_og_sin.y*synth_atribute_np[i][1] 

xmax = xmin + 2
#xmax = 6
x = np.linspace(xmin, xmax, num=100, endpoint=True)

plt.plot(x, df_og.y, label="og", color="red")  
for i in range(1000,2000):
    # x_n = synth_feature_np[i,:,0]
    # x = (x_n+1)*((X_MAX - X_MIN)/2) + X_MIN
    y = synth_feature_np[i,:,0]
    
    #df_synth_renormed = pd.DataFrame(np.array([y]).T, columns = ["y"])

    # xmin_tmp = df_synth_renormed.x.min()
    # xmax_tmp = df_synth_renormed.x.max()

    # x_synth_corrected = np.linspace(xmin_tmp, xmax_tmp, num=len(x))
    # df_synth_corrected = df_synth_renormed.copy()
    # df_synth_corrected.loc[:,"x"] = x_synth_corrected
    
    # xmin_tmp = df_synth_renormed.x.min()
    # xmax_tmp = df_synth_renormed.x.max()

    # x_synth_corrected = np.linspace(xmin_tmp, xmax_tmp, num=len(x))
    # df_synth_corrected = df_synth_renormed.copy()
    # df_synth_corrected.loc[:,"x"] = x_synth_corrected
    
    plt.plot(x, y, label="synth", color="black", alpha=0.04)   
  
plt.show()
    