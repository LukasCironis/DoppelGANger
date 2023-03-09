import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gan.output import Output, OutputType, Normalization


def trig(type, freq, amp=1, 
         n=1005, xmin=-500):
    
    xmax = xmin + 2
    by = (xmax - xmin)/n
    x = np.arange(xmin, xmax, by)
    if type == "sin":
        y = amp*np.sin(x*freq + shift)
    elif type == "cos":
        y = amp*np.cos(x*freq + shift)
    else:
        raise NotImplementedError("'type' is only sin or cos")
    
    df=pd.DataFrame(np.array([y[5:]]).T, columns=["y"])
    df.loc[:,"type"] = type
    df.loc[:,"freq"] = freq
    # df.loc[:,"shift"] = shift
    # df.loc[:,"xmin"] = x[5]
    
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
    
    
if __name__ == "__main__":
    
    fig, axes = plt.subplots(1,1)

    for i in range(5):
           
        pars = {
            "type" : "cos",
            "freq" : [0.5, 1, 2, 5][i],
            "amp"  : 1,  
            "xmin": -5, 
            "n" : 105
        }
        
        df_test = trig(**pars)
        df_test.y.plot(ax=axes)
    
    plt.show()
    
    df_list = []
    feature_list = []
    atribute_list = []

    X_MIN = -6
    X_MINMAX = 4
    X_MAX = X_MINMAX + 2
    FREQ_MAX = 6
    FREQ_MIN = 0
    SHIFT_MAX = 7
    SHIFT_MIN = 0
    
    for _ in range(100):
        type = np.random.choice(["cos", "sin"])
        freq = 6*np.random.random_sample(1)[0]
        #shift = 7*np.random.random_sample(1)[0]
        xmin = np.random.uniform(X_MIN, X_MINMAX)
        xmax = xmin + 2
        tmp = trig(type, freq, xmin=xmin, n=100+5)
        
        #normalize        
        freq = (freq - FREQ_MIN)/(FREQ_MAX - FREQ_MIN)
        #shift = (shift - SHIFT_MIN)/(SHIFT_MAX - SHIFT_MIN)
        #xmin = 2*(xmin - X_MIN)/(X_MINMAX - X_MIN) - 1
        #xmax = 2*(xmax - X_MIN)/(X_MINMAX - X_MIN) - 1

        #tmp.loc[:, "x"] = 2*(tmp.loc[:, "x"]-X_MIN)/(X_MAX-X_MIN) - 1
        tmp.loc[:, "freq"] = freq
        #tmp.loc[:, "shift"] = shift
        #tmp.loc[:, "xmin"] = xmin

        #tmp.iloc[:,-5:] = 2*(tmp.iloc[:,-5:]-X_MIN)/(X_MAX-X_MIN) - 1
        
        df_list.append(tmp)
        atribute_list.append([
            1 if type=="cos" else 0,
            1 if type=="sin" else 0, 
            freq, 
            #shift, 
            #xmin, 
            ] + list(tmp.iloc[0,-5:]))
        
        feature_list.append(tmp[["y"]].values)
    
    df = pd.concat(df_list).reset_index(drop=True) 
    df.type.value_counts()
    
    features_np = np.array(feature_list)
    features_np.shape
    features_np.min()
    features_np.max()
 
    atribute_np = np.array(atribute_list)
    atribute_np.shape
    atribute_np.min()
    atribute_np.max()
    
    data_gen_flag = np.ones((features_np.shape[0],features_np.shape[1]))
    data_gen_flag.shape
    
    data_feature_output = [
        #Numeric
        Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
        #Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
    ]
        
    data_attribute_output = [
        Output(type_=OutputType.DISCRETE, dim=2, normalization=None, is_gen_flag=False),
        Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
        Output(type_=OutputType.CONTINUOUS, dim=5, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
        #Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
        #Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
    ]

    with open('data_feature_output.pkl', 'wb') as f:
        pickle.dump(data_feature_output, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_attribute_output.pkl', 'wb') as f:
        pickle.dump(data_attribute_output, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.savez("data_train.npz", 
        data_feature = features_np.astype(np.float32),
        data_attribute = atribute_np.astype(np.float32),
        data_gen_flag = data_gen_flag.astype(np.float32)
        )
    
    #visualize
    for i in range(6,10):
        plt.plot(features_np[i])
    
    plt.show()
    