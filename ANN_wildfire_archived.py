import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet') 
from keras.models import load_model
#from keras.optimizers import Adam
from scipy.io import loadmat
import tensorflow as tf
import joblib
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import seaborn as sns

mydir = 'version1/'
# load in data
# 2 US, 5 SHSA, 8 NHAF, 9 SHAF
ann_output = []
data_output = []
for i in range(14):
    filename = mydir+'wildfire_surrogate'+str(i+1)+'.mat'
    tmp = loadmat(filename)
    ELMX = tmp['ELMX']
    ELMy = tmp['ELMy']
    
    # remove modeled burn area outlier
    #threshold = np.percentile(ELMy,99)
    #for j in range(len(ELMy)):
    #    if ELMy[j] >= threshold:
    #        ELMy[j] = threshold
    #ELMX = ELMX[~np.isnan(ELMX).any(axis=1)]
    #ELMy = ELMy[~np.isnan(ELMy).any(axis=1)]
    
    # rescaling
    sc_X = MinMaxScaler(feature_range=(0,1))
    sc_y = MinMaxScaler(feature_range=(0,1))

    scaler_filename = mydir+"scaler_X"+str(i+1)+'.mat'
    joblib.dump(sc_X, scaler_filename)
    scaler_filename = mydir+"scaler_y"+str(i+1)+'.mat'
    joblib.dump(sc_y, scaler_filename)
    
    X = sc_X.fit_transform(ELMX)
    y = sc_y.fit_transform(ELMy.reshape(-1,1))
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    # build fully-connected artifitial neural network
    ann = Sequential()
    ann.add(Dense(input_dim=17, output_dim = 5,activation='softplus',init='uniform'))
    ann.add(Dense(output_dim = 5,activation='softplus',init='uniform'))
    ann.add(Dense(output_dim = 5,activation='softplus',init='uniform'))
    ann.add(Dense(output_dim = 5,activation='softplus',init='uniform'))
    ann.add(Dense(output_dim = 5,activation='softplus',init='uniform'))
    ann.add(Dense(output_dim = 1,activation='softplus',init='uniform'))
    ann.summary()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    #ann.compile(optimizer=opt,loss= 'mse')
    #ann.fit(X_train,y_train,batch_size=20,nb_epoch=30)
    #ann.save(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus')
    
    ann = load_model(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus')
    
    y_pred = ann.predict(X)
    
    #ann_y =  np.vstack((sc_y.inverse_transform(y_pred_train1.reshape(-1,1)),sc_y.inverse_transform(y_pred_test1.reshape(-1,1))))
    #data_y  = np.vstack((sc_y.inverse_transform(y_train.reshape(-1,1)),sc_y.inverse_transform(y_test.reshape(-1,1))))
    ann_y =  sc_y.inverse_transform(y_pred.reshape(-1,1)).reshape(-1,360)
    data_y  = sc_y.inverse_transform(y.reshape(-1,1)).reshape(-1,360)
    #plt.scatter(ann_y,data_y)

    ann_output.append(np.sum(ann_y,0))
    data_output.append(np.sum(data_y,0))
    
ann_output, data_output = np.array(ann_output), np.array(data_output)
pd.DataFrame(ann_output).to_csv(mydir+"ann_output.csv")
pd.DataFrame(data_output).to_csv(mydir+"data_output.csv")

ann_output = pd.read_csv(mydir+"ann_output.csv")
data_output = pd.read_csv(mydir+"data_output.csv")
ann_output = ann_output.iloc[:,:-1].values
data_output = data_output.iloc[:,:-1].values

##-------------------------- tuning of surrogate wildfire model 
ann_output_tune = []
data_output_tune = []
for i in range(14):
    filename = mydir+'wildfire_surrogate'+str(i+1)+'.mat'
    tmp = loadmat(filename)
    OBSX = tmp['OBSX']
    OBSy = tmp['OBSy']
    
    # remove observed burn area outlier
    #threshold = np.percentile(OBSy,99)
    #for j in range(len(OBSy)):
    #    if OBSy[j] >= threshold:
    #        OBSy[j] = threshold
    #OBSX = OBSX[~np.isnan(OBSX).any(axis=1)]
    #OBSy = OBSy[~np.isnan(OBSy).any(axis=1)]

    # rescaling
    sc_X = joblib.load(mydir+"scaler_X"+str(i+1)+'.mat')
    sc_y = joblib.load(mydir+"scaler_y"+str(i+1)+'.mat')
    
    X = sc_X.fit_transform(OBSX)
    y = sc_y.fit_transform(OBSy.reshape(-1,1))
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    ann = load_model(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus')
    '''
    if i==4 or i == 7 or i == 8:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=2000,
        decay_rate=0.995)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        ann.compile(optimizer=opt,loss= 'mse')
        ann.fit(X_train,y_train,batch_size=20,nb_epoch=100)
        ann.save(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus_tuned3')
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        ann.compile(optimizer=opt,loss= 'mse')
        ann.fit(X_train,y_train,batch_size=20,nb_epoch=100)
        ann.save(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus_tuned2')
    '''
    if i==7 or i==8:
        ann = load_model(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus_tuned3')
    else:
        ann = load_model(mydir+'wildfire_surrogate'+str(i+1)+'_ANN_softplus_tuned2')
    
    y_pred = ann.predict(X)
    
    ann_y =  sc_y.inverse_transform(y_pred.reshape(-1,1)).reshape(-1,120)
    data_y  = sc_y.inverse_transform(y.reshape(-1,1)).reshape(-1,120)

    ann_output_tune.append(np.sum(ann_y,0))
    data_output_tune.append(np.sum(data_y,0))

ann_output_tune, data_output_tune = np.array(ann_output_tune), np.array(data_output_tune)
pd.DataFrame(ann_output_tune).to_csv(mydir+"ann_output_tune.csv")
pd.DataFrame(data_output_tune).to_csv(mydir+"data_output_tune.csv")

ann_output = np.array(pd.read_csv(mydir+'ann_output.csv')) # surrogate
data_output = np.array(pd.read_csv(mydir+'data_output.csv')) # ELM
ann_output_tune = np.array(pd.read_csv(mydir+'ann_output_tune.csv')) # tuned surrogate
data_output_tune = np.array(pd.read_csv(mydir+'data_output_tune.csv')) # obs

