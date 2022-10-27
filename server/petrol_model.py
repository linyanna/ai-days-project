#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
import os

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import sklearn
import time

import matplotlib.pyplot as plt


# In[5]:


import sys

sys.path.insert(1, './common/')
import utils
import plots


# In[6]:


# global parameters to control behavior of the pre-processing, ML, analysis, etc.
seed = 42

# deterministic seed for reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)

prop_vec = [24, 2, 2]


# In[7]:


petrol = pd.read_csv("./cars_data_1/petrol.csv")
petrol.head()


# In[8]:


petrol.info()


# In[9]:


petrol.fuel.value_counts()


# In[36]:


petrol_all_xy = petrol.to_numpy()

col_names = [c for c in petrol.columns]
features = col_names[3:5]
target = col_names[5]


# In[37]:


print('features: {} --- target: {}'.format(features, target))


# In[ ]:





# In[38]:


petrol_all_x = petrol_all_xy[:,3:5]
petrol_all_y = petrol_all_xy[:,5]


# In[39]:


assert petrol_all_x.shape == (6390, 2)


# In[40]:


petrol_all_y.shape


# In[41]:


# min-max normalize the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(copy=True)
scaler.fit(petrol_all_x) 

scaled_petrol_all_x = scaler.transform(petrol_all_x)


# In[42]:


# grab the data
petrol_train_x, petrol_train_y, petrol_test_x, petrol_test_y, petrol_val_x, petrol_val_y, petrol_all_x, petrol_all_y = utils.load_preprocess_mnist_data(onehot=True, prop_vec=prop_vec, seed=seed)

# sanity check shapes
petrol_train_x.shape, petrol_train_y.shape, petrol_test_x.shape, petrol_test_y.shape, petrol_val_x.shape, petrol_val_y.shape


# In[43]:


# Let's create a custom callback class
class PerfEvalCustomCallback(keras.callbacks.Callback):
    
    def __init__(self, perf_data):
        self.perf_data = perf_data
    
    # we define the on_epoch_end callback and save the loss and accuracy in perf_data
    def on_epoch_end(self, epoch, logs=None):
        self.perf_data[epoch,0] = logs['loss']
        self.perf_data[epoch,1] = logs['accuracy']
        self.perf_data[epoch,2] = logs['val_loss']
        self.perf_data[epoch,3] = logs['val_accuracy']

    def get_perf_data():
        return self.perf_data


# In[44]:


# Plot the model's performance during training (across epochs)
def plot_training_perf(train_loss, train_acc, val_loss, val_acc, fs=(8,5)):
    plt.figure(figsize=fs)


    assert train_loss.shape == val_loss.shape and train_loss.shape == val_acc.shape and val_acc.shape == train_acc.shape
    
    # assume we have one measurement per epoch
    num_epochs = train_loss.shape[0]
    epochs = np.arange(0, num_epochs)
    
    # Can you figure out why this makes sense? Why remove -0.5?
    plt.plot(epochs-0.5, train_loss, 'm', linewidth=2,  label='Loss (Training)')
    plt.plot(epochs-0.5, train_acc, 'r--', linewidth=2, label='Accuracy (Training)')
    
    plt.plot(epochs, val_loss, 'g', linewidth=2, label='Loss (Validation)')
    plt.plot(epochs, val_acc, 'b:', linewidth=2, label='Accuracy (Validation)')
    
    
    plt.xlim([0, num_epochs])
    plt.ylim([0, 1.05])
    
    plt.legend()
    
    plt.show()


# In[45]:


# Customize this function as you like but makes sure it is implemented correctly.    
# Note: If you need to change the method definition to add more arguments, make sure to make 
# the new arguments optional (and have a sensible default value)
def evaluate_model(name, model, eval_data, 
                   plot_training=True, evaluate_on_test_set=True):
    
    # unpack the stuff
    perf_data, dataset = eval_data
    train_x, train_y, val_x, val_y, test_x, test_y = dataset
    
    # get predictions from the model
    train_preds = model.predict(train_x)
    val_preds = model.predict(val_x)
    
    # measure the accuracy (as categorical accuracy since we have a softmax layer)
    catacc_metric = keras.metrics.CategoricalAccuracy()
    catacc_metric.update_state(train_y, train_preds)
    train_acc = catacc_metric.result()
    
    catacc_metric = keras.metrics.CategoricalAccuracy()
    catacc_metric.update_state(val_y, val_preds)
    val_acc = catacc_metric.result()
    print('[{}] Training Accuracy: {:.3f}%, Validation Accuracy: {:.3f}%'.format(name, 100*train_acc, 100*val_acc))
    
    if plot_training:
        plot_training_perf(perf_data[:,0], perf_data[:,1], perf_data[:,2], perf_data[:,3])
        
    if evaluate_on_test_set:
        ### Evaluate the model on test data and put the results in 'test_loss', 'test_acc' (set verbose = 0)
        ###* put your code here (~1-2 lines) *###
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
        
        print('[{}]  Test loss: {:.5f}; test accuracy: {:.3f}%'.format(name, test_loss, 100*test_acc))
        
    
    # You can add stuff here if you want
    ###* put your code here (0+ lines) *###
    
    
    return

# this is what we call to do the training
# def train_model(model, max_epochs=25, batch_size=100, verbose=0, 
#                    dataset=(train_x, train_y, val_x, val_y, test_x, test_y)):
def train_model(model,dataset, max_epochs=25, batch_size=100, verbose=0):

    # unpack dataset
    train_x, train_y, val_x, val_y, test_x, test_y = dataset
    
    # this is the callback we'll use for early stopping
    early_stop_cb = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=4)
    
    # setup the performance data callback
    perf_data = np.zeros((max_epochs, 4))
    perf_eval_cb = PerfEvalCustomCallback(perf_data)
    
    hobj = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=max_epochs, batch_size=batch_size, 
                     shuffle=True, callbacks=[perf_eval_cb, early_stop_cb], verbose=verbose)
    
    eff_epochs = len(hobj.history['loss'])
    eval_data = (perf_data[0:eff_epochs,:], dataset) # tuple of evaluation data
    
    return eval_data


# In[46]:


def create_compile_model0(fixed, input_shape=784, num_outputs=10, verbose=True):
    name = 'Model0--Fixed' if fixed else 'Model0--Broken'
    hidden_widths=[300, 100]
    
    model = keras.models.Sequential(name=name)
    
    model.add(keras.Input(shape=(input_shape,), sparse=False)) 
    
    for i, hw in enumerate(hidden_widths):
        model.add(keras.layers.Dense(hw, activation='relu', name='hidden_{}'.format(i), 
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(1/hw)),
                                     bias_initializer=keras.initializers.Zeros()))
        
    model.add(keras.layers.Dense(num_outputs, activation='softmax', name='output',
                                kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(0.1)),
                                bias_initializer=keras.initializers.Zeros()))
    
    opt = keras.optimizers.Adam(learning_rate=0.0025)
    
    if verbose:
        model.summary()
    
    if fixed:
        ###* put your code here (~1-2 lines) *###
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # comment/remove this line once you implement the fix
        # raise NotImplementedError 
        
    else:
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    
    return name, model


# In[47]:


## create and compile the model for fixed=True, train it, then evaluate it
name, model = create_compile_model0(True, verbose=False) 

petrol_dataset = petrol_train_x, petrol_train_y, petrol_val_x, petrol_val_y, petrol_test_x, petrol_test_y
# train
eval_data = train_model(model, petrol_dataset)

# evaluate
evaluate_model(name, model, eval_data)


# In[55]:


# predictions = model.predict(petrol_train_x)
# plt.scatter(petrol_train_y, predictions)


# In[57]:


import pickle


# In[58]:


filename = 'petrol_model'
pickle.dump(model,open(filename, 'wb'))


# In[ ]:




