# By David Boe based on tutorial in towardsdatascience.com

# The usual collection of indispensables 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.fftpack

# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras

# Time series prediction model
def dnn_keras_tspred_model():
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam() # Default learning rate = 0.001
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae']) 
  model.summary()
  return model

num_train_data = 4000
num_test_data = 1000
timestep = 0.1
tm =  np.arange(0, (num_train_data+num_test_data)*timestep, timestep)
y = np.sin(tm) + np.sin(tm*np.pi/2) + np.sin(tm*(-3*np.pi/2)) 
SNR = 10
ypn = y + np.random.normal(0,10**(-SNR/20),len(y))

plt.plot(tm[0:100],y[0:100], label='Signal')
plt.plot(tm[0:100],ypn[0:100],'r', label='Noisy signal') # red one is the noisy signal
plt.legend()
plt.grid()
plt.title('Time series signal')

# prepare the train_data and train_labels
dnn_numinputs = 64
num_train_batch = 0
train_data = []
for k in range(num_train_data-dnn_numinputs-1):
  train_data = np.concatenate((train_data,ypn[k:k+dnn_numinputs]));
  num_train_batch = num_train_batch + 1  
train_data = np.reshape(train_data, (num_train_batch,dnn_numinputs))
train_labels = y[dnn_numinputs:num_train_batch+dnn_numinputs]
print(y.shape, train_data.shape, train_labels.shape)

model = dnn_keras_tspred_model()
EPOCHS = 100
strt_time = datetime.datetime.now()
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                  validation_split=0.2, verbose=0,
                  callbacks=[])
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
plt.grid()
plt.title('Val loss vs. epoch')

# test how well DNN predicts now
num_test_batch = 0
strt_idx = num_train_batch
test_data=[]
for k in range(strt_idx, strt_idx+num_test_data-dnn_numinputs-1):
  test_data = np.concatenate((test_data,ypn[k:k+dnn_numinputs]));
  num_test_batch = num_test_batch + 1  
test_data = np.reshape(test_data, (num_test_batch, dnn_numinputs))
test_labels = y[strt_idx+dnn_numinputs:strt_idx+num_test_batch+dnn_numinputs]


dnn_predictions = model.predict(test_data).flatten()
keras_dnn_err = test_labels - dnn_predictions
plt.plot(dnn_predictions[0:100], label='DNN predictions')
plt.plot(test_labels[0:100],'r', label='Test labels')
plt.grid()
plt.legend()
plt.title('DNN predictions and test labels')

#LMS
M = 1000
L = 64
yrlms = np.zeros(M+L)
#wn = np.random.normal(0,1,L)
wn = np.zeros(L)
print(wn.shape, yrlms.shape)
mu = 0.005
for k in range(L,M+L):
  yrlms[k] = np.dot(ypn[k-L:k],wn)
  e = ypn[k]- yrlms[k]
  wn=wn+(mu*ypn[k-L:k]*e)

plt.plot(yrlms[600:800], label='yrlms')
plt.plot(y[600:800],'r',label='Signal')
plt.grid()
plt.legend()
plt.title('LMS Results')

dnn_err = dnn_predictions - test_labels
lms_err = yrlms[0:M] - y[0:M]
plt.plot(dnn_err,label='DNN errors')
plt.plot(lms_err,'r', label='LMS errors')
plt.grid()
plt.legend()
plt.title('DNN vs. LMS results')
plt.show()

dnn_mse = 10*np.log10(np.mean(pow(np.abs(dnn_err),2)))
lms_mse = 10*np.log10(np.mean(pow(np.abs(lms_err[200:M]),2)))
lms_sigpow = 10*np.log10(np.mean(pow(np.abs(y[0:M]),2)))
dnn_sigpow = 10*np.log10(np.mean(pow(np.abs(test_labels),2)))

dnn_sigpow_var = 10*np.log10(np.var(test_labels))
print(dnn_sigpow)
print(dnn_sigpow_var)

#print(dnn_mse, dnn_sigpow, lms_mse, lms_sigpow)
print("Neural network SNR:", dnn_sigpow - dnn_mse)
print("LMS Prediction SNR:", lms_sigpow - lms_mse)


