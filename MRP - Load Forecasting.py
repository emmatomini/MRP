
# coding: utf-8

# **MRP - Load Prediction using Recurrent Neural Networks for Ontario Electricity Market**
# 
# Emmalie Tomini
# 
# Ryerson University
# 
# The following program uses a recurrent neural network structure to test which weather variables are most important for creating load forecast predictions 24 hours in to the future for the Ontario Electricity Market.

# In[69]:


# Import Libraries
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
get_ipython().run_line_magic('matplotlib', 'inline')
import shutil 
from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from tensorflow.python.keras.initializers import RandomUniform
import h5py
tf.__version__


# Import the dataset "2008-2017.xlsx", made up of seasonal and weather data collected from Toronto Buttonville Airport weather station.  This dataset contains several weather variables, some of which are more interesting than others.  We will train the RNN on various combinations of features to determine which are necessary to build the most accurate model.

# In[47]:


# import load data
load_data = "C:/Users/emmat_000/Desktop/School/Data Science/MRP/2008-2017Load.xlsx"
sheet = "2008-2017Load"
df = pd.read_excel(io=load_data, sheet_name=sheet)


# Below, we plot "Total Market Demand" over the entire dataset.

# In[48]:


df[['Total Market Demand']].plot(c='b', title='Total Market Demand (2008-2017)')


# The x_df dataframe contains our input feature variables.  These include the year, the day, the hour, what day of the week it is, the temperature in degrees celcius, the relative humidity and the dew point temperature.  Select the corresponding features for the appropriate model.

# In[49]:


df.fillna(0, inplace=True)
x_df = df[['Year', 'Day365', 'Hour', 'Weekday', 'Temp (°C)', 'Rel Hum (%)', 'Dew Point Temp (°C)']]
y_df = df[['Total Market Demand']]


# Our data is constructed in hourly intervals.  Our initial experiment will be to predict the energy demand 24 hours in to the future.  If we want to predict even further in to the future, the day and time step variables are set up to allow such alteration for future experiments.  However, for the time being we will predict electricity load 24 hours in to the future by shifting our output feature, total market demand, 24 steps back.

# In[1]:


shift_days = 1
shift_steps = shift_days*24  # Number of hours


# Shift our target variable.  We want to predict the future so we shift the y variables up.

# In[3]:


y_df = y_df.shift(-shift_steps)


# From the previous literature, scalig the variables between 0 and 1 was found t0 increase convergence. This method will be applied the the data using the MinMaxScaler function, default is already 0-1

# In[55]:


x_df[len(x_df)-6:]


# In[56]:


# save an unaltered copy of y_df
y_original = y_df


# In[57]:


# Scale x_df
scale_x = MinMaxScaler(copy=False)
scale_x.fit(x_df)
x_df = scale_x.transform(x_df)

# Scale y_df
scale_y = MinMaxScaler(copy=False)
scale_y.fit(y_df)
y_df = scale_y.transform(y_df)


# Seperate dataset in to training and testing

# In[85]:


train_len = int(0.9*(len(x_df)))

x_train = x_df[0:train_len]
y_train = y_df[0:train_len]
y_train_o = y_original[0:train_len] # not rescaled
 
x_test = x_df[train_len:]
y_test = y_df[train_len:]
y_test_o = y_original[train_len:] # not rescaled


# Initialize hyper parameters: s_length = sequence length fed to the model for training, b_size = number of sequences, num_f = number of input features

# In[60]:


# Sequrnce length of approximatly 2 months , 8 weeks
s_length = 24*7*8  

# Batch size needs to be adjusted in order to acheive best CPU operation
# this value can be adjusted later during testing and experiments
b_size = 150

# During experiments, we may decide to add or remove features to increase
# accuracy.  We will specify the number of x features we are using for each
# experiment and this will be passed as an argument to the batch generator
num_f = 7


# Next, we generate atches of sequences, rather than feeding the entire training set sequence in to the algorithm.  Here we can make use of Keras functions to ease the process.

# In[61]:


def batchGenerator(b_size, s_length):
    
    while True:
        
            x_dim = (b_size, s_length, num_f)
            x_batch = np.zeros(x_dim)
            
            y_dim = (b_size, s_length, 1)
            y_batch = np.zeros(y_dim)
            
            for i in range(b_size):
                j = np.random.randint(train_len - s_length)
                x_batch[i] = x_train[j:(j+s_length)]
                y_batch[i] = y_train[j:(j+s_length)]
            yield(x_batch, y_batch)


# We call our bath generator and create **150** batches, each containing a sequence of 1344 observations.  We verify the structure of our batches by printing their sizes.

# In[63]:


gen_batches = batchGenerator(b_size, s_length)
x_batch, y_batch = next(gen_batches)
print(x_batch.shape)
print(y_batch.shape)


# In[64]:


# plot one of the output sequences to check
plt.plot(y_batch[6])


# Now, we are prepared to implement our RNN model

# In[65]:


# During testing, the optimal number of hidden units can be adjusted
model = Sequential()
model.add(GRU(units=200,
              return_sequences=True,
              input_shape=(None, num_f,)))
model.add(Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(-0.01,0.01)))


# Compile Time!!!!

# In[70]:


# define optimization function
adam_opt = Adam(lr=0.001, beta_1=0.85, beta_2=0.95, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="mse", optimizer=adam_opt, metrics=['mae', 'acc'])


# In[71]:


model.summary()


# In[72]:


valid_data = (np.expand_dims(x_test, axis=0),np.expand_dims(y_test, axis=0))


# We define callback functions to define checkpoints and log progress.
# 
# The following callback saves the model after each epoc.

# In[73]:


checkpoint_callback = ModelCheckpoint('model_temp_dew_hum.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, period=1)


# When the performence worsens, the following callback will implement early stopping

# In[74]:


earlyStopping_callback = EarlyStopping(monitor='val_loss', patience=3)


# Write to tensorboard callback

# In[75]:


tensorboard_callback  = TensorBoard(log_dir='./mrp_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# Models often benefit from reducing the learning rate once learning stagnates.  The following callback reduces the learning rate if loss hasnot improved in the last epoch by multiplying  it by a factor of 2.

# In[76]:


reduceLR_callback = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       min_lr=0.0001,
                                       patience=0)


# In[77]:


callbacks = [checkpoint_callback,
             earlyStopping_callback,
             tensorboard_callback,
             reduceLR_callback]


# Train the model based on the appropriately selected input features.
# 
# If you do not intent on training the model, skip to the next line and load the weights from a previously trained model.

# In[78]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(generator=gen_batches,\n                    epochs=20,\n                    steps_per_epoch=100,\n                    validation_data=valid_data,\n                    callbacks=callbacks)')


# To load the model weights from a trained model, select the appropriate H5 file.

# In[79]:


model.load_weights('model_temp_dew_hum.h5')


# Performance on test set, we run the test set in as one long batch

# In[80]:


result = model.evaluate(x=np.expand_dims(x_test,axis=0), y=np.expand_dims(y_test, axis=0))


# In[81]:


print("loss (test-set):", result)


# Now we are able to make predictions with our model.
# 
# First we run the entire test set through the model and gather the MAPE score.

# In[4]:


# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


x = np.expand_dims(x_test, axis=0)
y_pred = model.predict(x)
y_rescale = y_pred[0,:,0]
y_rescale=np.expand_dims(y_rescale, axis=1)
y_pred_rescaled = scale_y.inverse_transform(y_rescale)
y_true = y_test_o
mape_result = mean_absolute_percentage_error(y_true[200:], y_pred_rescaled[200:])
mape_result


# Plot the prediction results over the entire test set.

# In[ ]:


#plot actual VS predicted
plt.figure(figsize=(15,5))
plt.plot(y_true, label='true')
plt.plot(y_pred_rescaled, label='pred')
plt.ylabel("Total Market Demand")
plt.legend()
plt.show()


# Change the length of the test set sequence fed to the model for specific predictions.  For example, below we give the model two weeks worth of input data and plot the last 24 data points, that is, we are making a day ahead load forcast for October 2nd, 2017.

# In[6]:


# make prediction
x = np.expand_dims(x_test[6100:6700], axis=0)
y_pred = model.predict(x)
y_rescale = y_pred[0,:,0]
y_rescale=np.expand_dims(y_rescale, axis=1)
y_pred_rescaled = scale_y.inverse_transform(y_rescale)
y_true = y_test_o[6100:6700]


# In[7]:


# plot result
#plot actual VS predicted
plt.figure(figsize=(15,5))
plt.plot(y_true[467:491], label='true')
plt.plot(y_pred_rescaled[467:491], label='pred')
plt.ylabel("Total Market Demand")
plt.legend()
plt.show()


# In[ ]:


# calculate MAPE
mape_result = mean_absolute_percentage_error(y_true[467:491], y_pred_rescaled[467:491])
mape_result

