import numpy as np
import pickle
import time
#keras/sklearn libraries
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Input, Reshape, BatchNormalization
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, AveragePooling1D, Flatten, Concatenate
from keras import backend 
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


###monitor program run time
start_time = time.time()

###check what device code is running on (could be relevant for GPU)
print(device_lib.list_local_devices())

###open dataset in order: (1) dos of surface, (2) adsorption energy(target), (3) surface elements/directory of file - for checking, (4) intrinsic atomic params - not currently used, (5) dos of adsorbate in gas phase (only for general case)
###run_combined=0 runs ML for each adsorbate separately, =1 runs ML for all adsorbates simultaneously; a slightly modified model is used here
run_combined=1

if run_combined == 0:
  with open('CH_data', 'rb') as f:
    dos_x_temp = pickle.load(f) 
    energy_y_temp = pickle.load(f)
    metadata_temp = pickle.load(f)
    params_x_temp = pickle.load(f)
elif run_combined == 1:
  with open('Combined_data', 'rb') as f:
    dos_x_temp = pickle.load(f) 
    energy_y_temp = pickle.load(f)
    metadata_temp = pickle.load(f)
    params_x_temp = pickle.load(f)
    adsorbate_x_temp = pickle.load(f)


###Some data rearranging, depends on if atomic params are to be included as extra features in the DOS series or separately
###entries 1700-2200 of the data are set to zero, these are states far above fermi level which seem to cause additional errors, reason being some states are not physically reasonable

###First column is energy; not used in current implementation
dos_x_comb=dos_x_temp[:,0:2000,1:28]
###States far above fermi level can be unphysical and set to zero
dos_x_comb[:,1800:2000,0:27]=0
###float32 is used for memory concerns
dos_x_comb=dos_x_comb.astype(np.float32)

if run_combined == 1:
  adsorbate_x_comb=adsorbate_x_temp[:,0:2000,1:10]
  adsorbate_x_comb=adsorbate_x_comb.astype(np.float32)
  

###Creates the ML model with keras
###This is the generic model where all 3 adsorption sites are fitted at the same time
def create_model(shared_conv): 
  
  ###Each input represents one out of three possible bonding atoms
  input1= Input(shape=(2000,channels))   
  input2= Input(shape=(2000,channels)) 
  input3= Input(shape=(2000,channels)) 
  
  conv1=shared_conv(input1)
  conv2=shared_conv(input2)
  conv3=shared_conv(input3)
  
  convmerge=Concatenate(axis=-1)([conv1, conv2, conv3])
  convmerge=Flatten()(convmerge)
  convmerge=Dropout(0.2)(convmerge)
  convmerge=Dense(200, activation='linear')(convmerge)
  convmerge=Dense(1000, activation='relu')(convmerge)
  convmerge=Dense(1000, activation='relu')(convmerge)
  
  out=Dense(1, activation='linear')(convmerge)
  #shared_conv.summary()  
  model=Model(input=[input1, input2, input3], output=out)  
  return model

###This is the generic model where all 3 adsorption sites are fitted at the same time, and all adsorbates are fitted as well
def create_model_combined(shared_conv): 
  
  ###Each input represents one out of three possible bonding atoms
  input1= Input(shape=(2000,channels))   
  input2= Input(shape=(2000,channels)) 
  input3= Input(shape=(2000,channels))  
  input4= Input(shape=(2000,channels)) 
  
  conv1=shared_conv(input1)
  conv2=shared_conv(input2)
  conv3=shared_conv(input3)
  
  adsorbate_conv=adsorbate_dos_featurizer()
  conv4=adsorbate_conv(input4)
  
  convmerge=Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
  convmerge=Flatten()(convmerge)
  convmerge=Dropout(0.2)(convmerge)
  convmerge=Dense(200, activation='linear')(convmerge)
  convmerge=Dense(1000, activation='relu')(convmerge)
  convmerge=Dense(1000, activation='relu')(convmerge)

  out=Dense(1, activation='linear')(convmerge)
  
  model=Model(input=[input1, input2, input3, input4], output=out)  
  return model


###This sub-model is the convolutional network for the DOS
###Uses the same model for each atom input channel
###Input is a 2000 length DOS data series
def dos_featurizer():
  input_dos=Input(shape=(2000,channels)) 
  x1=AveragePooling1D(pool_size=4, strides=4,padding='same')(input_dos)
  x2=AveragePooling1D(pool_size=25, strides=4,padding='same')(input_dos)
  x3=AveragePooling1D(pool_size=200, strides=4,padding='same')(input_dos) 
  x=Concatenate(axis=-1)([x1,x2,x3])
  x= Conv1D(50, 20, activation='relu',padding='same',strides=2)(x) 
  x=BatchNormalization()(x)
  x= Conv1D(75, 3, activation='relu',padding='same',strides=2)(x) 
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(100, 3, activation='relu',padding='same',strides=2)(x)
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(125, 3, activation='relu',padding='same',strides=2)(x) 
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(150, 3, activation='relu',padding='same',strides=1)(x) 
  shared_model=Model(input_dos, x)
  return(shared_model)
###Uses the same model for adsorbate but w/ separate weights
def adsorbate_dos_featurizer():
  input_dos=Input(shape=(2000,channels)) 
  x1=AveragePooling1D(pool_size=4, strides=4,padding='same')(input_dos)
  x2=AveragePooling1D(pool_size=25, strides=4,padding='same')(input_dos)
  x3=AveragePooling1D(pool_size=200, strides=4,padding='same')(input_dos) 
  x=Concatenate(axis=-1)([x1,x2,x3])
  x= Conv1D(50, 20, activation='relu',padding='same',strides=2)(x) 
  x=BatchNormalization()(x)
  x= Conv1D(75, 3, activation='relu',padding='same',strides=2)(x) 
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(100, 3, activation='relu',padding='same',strides=2)(x)
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(125, 3, activation='relu',padding='same',strides=2)(x) 
  x=AveragePooling1D(pool_size=3, strides=2,padding='same')(x)
  x= Conv1D(150, 3, activation='relu',padding='same',strides=1)(x) 
  shared_model=Model(input_dos, x)
  return(shared_model)


###Learning rate scheduler
###TODO: make better one with intelligent steps
def decay_schedule(epoch, lr):
    if epoch == 0:
        lr = 0.001
    elif epoch == 15:
        lr = 0.0005
    elif epoch == 35:
        lr = 0.0001
    elif epoch == 45:
        lr = 0.00005
    elif epoch == 55:
        lr = 0.00001
    return lr

###Some parameters
batch_size = 32
if run_combined == 1:
  batch_size = 128

epochs = 60
channels=9
seed=10
split_ratio=0.2

###Split data into train and test
if run_combined == 0:
  x_train, x_test, y_train, y_test, metadata_train, metadata_test, params_train ,params_test = train_test_split(dos_x_comb, energy_y_temp, metadata_temp, params_x_temp, test_size=split_ratio, random_state=88)
elif run_combined == 1:
  x_train, x_test, y_train, y_test, metadata_train, metadata_test, params_train ,params_test,ads_train, ads_test = train_test_split(dos_x_comb, energy_y_temp, metadata_temp, params_x_temp, adsorbate_x_comb, test_size=split_ratio, random_state=88)            
###Scaling data 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[2])).reshape(x_test.shape)
params_train = scaler.fit_transform(params_train)
params_test = scaler.transform(params_test)

if run_combined == 1:
  ads_train = scaler.fit_transform(ads_train.reshape(-1, ads_train.shape[2])).reshape(ads_train.shape)
  ads_test = scaler.transform(ads_test.reshape(-1, ads_test.shape[2])).reshape(ads_test.shape)

###call and fit model
shared_conv = dos_featurizer()
shared_conv2 = dos_featurizer()
lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)
lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), histogram_freq=1)

###FOr testing purposes, a model where 3 adsorption sites fitted simultaneously and 3 separately are done by comparison
if run_combined == 0:
  model = create_model(shared_conv)
  model.compile(loss='logcosh', optimizer=Adam(0.001),metrics=['mean_absolute_error'])
  model.summary()
  model.fit([x_train[:,:,0:9],x_train[:,:,9:18],x_train[:,:,18:27]], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([x_test[:,:,0:9],x_test[:,:,9:18],x_test[:,:,18:27]], y_test), callbacks=[tensorboard, lr_scheduler])
  train_out=model.predict([x_train[:,:,0:9],x_train[:,:,9:18],x_train[:,:,18:27]])
  train_out=train_out.reshape(len(train_out))
  test_out=model.predict([x_test[:,:,0:9],x_test[:,:,9:18],x_test[:,:,18:27]])
  test_out=test_out.reshape(len(test_out))

elif run_combined == 1:
  model = create_model_combined(shared_conv)
  model.compile(loss='logcosh', optimizer=Adam(0.001),metrics=['mean_absolute_error'])
  model.summary()
  model.fit([x_train[:,:,0:9],x_train[:,:,9:18],x_train[:,:,18:27],ads_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([x_test[:,:,0:9],x_test[:,:,9:18],x_test[:,:,18:27],ads_test], y_test), callbacks=[tensorboard, lr_scheduler])
  train_out=model.predict([x_train[:,:,0:9],x_train[:,:,9:18],x_train[:,:,18:27],ads_train])
  train_out=train_out.reshape(len(train_out))
  test_out=model.predict([x_test[:,:,0:9],x_test[:,:,9:18],x_test[:,:,18:27],ads_test])
  test_out=test_out.reshape(len(test_out))
  
###this is just to write the results to a file
if run_combined == 0:
  print('combined train MAE: ', mean_absolute_error(y_train, train_out))
  print('combined train RMSE: ', mean_squared_error(y_train, train_out)**(.5))
  print('combined test MAE: ', mean_absolute_error(y_test, test_out))
  print('combined test RMSE: ', mean_squared_error(y_test, test_out)**(.5))
  
  with open('model_predict_train.txt', 'w') as f:
      np.savetxt(f, np.stack((y_train, train_out), axis=-1))
  with open('model_predict_test.txt', 'w') as f:
      np.savetxt(f, np.stack((y_test, test_out), axis=-1))
  with open('metadata_test.txt', 'w') as f:
      for i in range(0,len(metadata_test)):
          f.write(str(metadata_test[i]) + '\n')
  with open('metadata_train.txt', 'w') as f:
      for i in range(0,len(metadata_train)):
          f.write(str(metadata_train[i]) + '\n')

elif run_combined == 1:
  print('train MAE: ', mean_absolute_error(y_train, train_out))
  print('train RMSE: ', mean_squared_error(y_train, train_out)**(.5))
  print('test MAE: ', mean_absolute_error(y_test, test_out))
  print('test RMSE: ', mean_squared_error(y_test, test_out)**(.5))
  
  with open('model_predict_train_combined.txt', 'w') as f:
      np.savetxt(f, np.stack((y_train, train_out), axis=-1))
  with open('model_predict_test_combined.txt', 'w') as f:
      np.savetxt(f, np.stack((y_test, test_out), axis=-1))
  with open('metadata_test_combined.txt', 'w') as f:
      for i in range(0,len(metadata_test)):
          f.write(str(metadata_test[i]) + '\n')
  with open('metadata_train_combined.txt', 'w') as f:
      for i in range(0,len(metadata_train)):
          f.write(str(metadata_train[i]) + '\n')

###kfold validation
kfold_on =1
if kfold_on == 1:  
  cvscores = []
  count=0
  kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
  
  for train, test in kfold.split(dos_x_comb, energy_y_temp):

    scaler_CV = StandardScaler()
    dos_x_comb[train,:,:] = scaler_CV.fit_transform(dos_x_comb[train,:,:].reshape(-1, dos_x_comb[train,:,:].shape[-1])).reshape(dos_x_comb[train,:,:].shape)
    dos_x_comb[test,:,:] = scaler_CV.transform(dos_x_comb[test,:,:].reshape(-1, dos_x_comb[test,:,:].shape[-1])).reshape(dos_x_comb[test,:,:].shape)
    if run_combined == 1:
      adsorbate_x_comb[train,:,:] = scaler.fit_transform(adsorbate_x_comb[train,:,:].reshape(-1, adsorbate_x_comb[train,:,:].shape[-1])).reshape(adsorbate_x_comb[train,:,:].shape)
      adsorbate_x_comb[test,:,:] = scaler.transform(adsorbate_x_comb[test,:,:].reshape(-1, adsorbate_x_comb[test,:,:].shape[-1])).reshape(adsorbate_x_comb[test,:,:].shape)
    
    keras.backend.clear_session()
    shared_conv = dos_featurizer()
    if run_combined == 0:
      model_CV = create_model(shared_conv)
      model_CV.compile(loss='logcosh', optimizer=Adam(0.001),metrics=['mean_absolute_error'])
      model_CV.fit([dos_x_comb[train,:,0:9],dos_x_comb[train,:,9:18],dos_x_comb[train,:,18:27]], energy_y_temp[train],
                batch_size=batch_size,
                epochs=epochs,verbose=0, callbacks=[lr_scheduler])
      scores = model_CV.evaluate([dos_x_comb[test,:,0:9],dos_x_comb[test,:,9:18],dos_x_comb[test,:,18:27]], energy_y_temp[test], verbose=0)
      train_out_CV_temp=model_CV.predict([dos_x_comb[test,:,0:9],dos_x_comb[test,:,9:18],dos_x_comb[test,:,18:27]])
      train_out_CV_temp=train_out_CV_temp.reshape(len(train_out_CV_temp))
    elif run_combined == 1:
      model_CV = create_model_combined(shared_conv)
      model_CV.compile(loss='logcosh', optimizer=Adam(0.001),metrics=['mean_absolute_error'])
      model_CV.fit([dos_x_comb[train,:,0:9],dos_x_comb[train,:,9:18],dos_x_comb[train,:,18:27], adsorbate_x_comb[train,:,:]], energy_y_temp[train],
                batch_size=batch_size,
                epochs=epochs,verbose=0, callbacks=[lr_scheduler])
      scores = model_CV.evaluate([dos_x_comb[test,:,0:9],dos_x_comb[test,:,9:18],dos_x_comb[test,:,18:27],adsorbate_x_comb[test,:,:]], energy_y_temp[test], verbose=0)
      train_out_CV_temp=model_CV.predict([dos_x_comb[test,:,0:9],dos_x_comb[test,:,9:18],dos_x_comb[test,:,18:27], adsorbate_x_comb[test,:,:]])
      train_out_CV_temp=train_out_CV_temp.reshape(len(train_out_CV_temp))
    print((model_CV.metrics_names[1], scores[1]))
    cvscores.append(scores[1])
    if count == 0:
      train_out_CV=train_out_CV_temp
      test_y_CV=energy_y_temp[test]
      test_index=test
    elif count > 0:
      train_out_CV= np.append(train_out_CV, train_out_CV_temp)
      test_y_CV= np.append(test_y_CV, energy_y_temp[test])
      test_index= np.append(test_index, test)
    count=count+1
    metadata_CV=[ metadata_temp[i] for i in test_index]
  print((np.mean(cvscores), np.std(cvscores)))
  print(len(test_y_CV))
  print(len(train_out_CV))
  with open('CV_predict.txt', 'w') as f:
      np.savetxt(f, np.stack((test_y_CV, train_out_CV), axis=-1))    
  with open('CV_predict_metadata.txt', 'w') as f:
      for i in range(0,len(metadata_CV)):
          f.write(str(metadata_CV[i]) + '\n')

print("--- %s seconds ---" % (time.time() - start_time))