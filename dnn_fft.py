# By David Boe based on tutorial in towardsdatascience.com

# The usual collection of indispensables 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.fftpack

# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras

# FFT prediction model
def dnn_keras_fft_model():
  model = keras.Sequential([
    keras.layers.Dense(NFFT*2, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(NFFT*2, activation=tf.nn.relu),
    keras.layers.Dense(NFFT*2)
  ])
  optimizer = tf.keras.optimizers.Adam()
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

# 64 point FFT
N = 64

# Using the same noisy signal used for LMS
yf = scipy.fftpack.fft(ypn[0:N])
yyf = scipy.fftpack.fft(y[0:N])

# Let us remove noise, easy to do at the FFT output
#yc = np.zeros(N,dtype=complex)
#cidx = np.where(np.abs(yf)>(N*0.2/2))[0]
#yc[cidx]=yf[cidx]

# 0 to Fs/2, Fs = 1/Ts
xf = np.linspace(0.0, 1.0/(2*timestep), int(N/2))

#fig, ax = plt.subplots()
# Plotting only from 0 to Fs/2
#plt.plot(xf, 2.0/N * np.abs(yc[:N//2]),'r')
plt.plot(xf, 2.0/N * np.abs(yyf[:N//2]), label='fft(y)')
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label='fft(ypn)')
plt.grid()
plt.legend()
plt.title('FFT')

# Train the DNN for 64 point FFT
NFFT = N
num_train_batch = 1
num_batches = 10000
train_data = np.random.normal(0,1,(num_batches, NFFT*2))
train_labels = np.random.normal(0,1,(num_batches, NFFT*2))
model = dnn_keras_fft_model()
for k in range(num_train_batch):
  for el in range(num_batches):
    fftin = train_data[el,0::2] + 1j*train_data[el,1::2]
    train_labels[el,0::2]=scipy.fftpack.fft(fftin).real
    train_labels[el,1::2]=scipy.fftpack.fft(fftin).imag
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
  train_data = np.random.normal(0,1,(num_batches, NFFT*2))

fftin = np.zeros((1,2*NFFT))
fftin[:,0::2]=ypn[0:NFFT]
fftout = model.predict(fftin).flatten()
fftout = fftout[0::2] + 1j*fftout[1::2]
plt.plot(xf, 2.0/NFFT * np.abs(fftout[0:NFFT//2]),label='DNN FFT')
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),'r',label=('SciPy FFT'))
plt.grid()
plt.legend()
plt.title('Frequency spectrum: DNN vs. FFT')

# Predict with randon input data
test_data = np.random.normal(0,1,(1000, NFFT*2))
test_labels = np.random.normal(0,1,(1000, NFFT*2))
for el in range(1000):
  fftin = test_data[el,0::2] + 1j*test_data[el,1::2]
  test_labels[el,0::2]=scipy.fftpack.fft(fftin).real
  test_labels[el,1::2]=scipy.fftpack.fft(fftin).imag

dnn_out = model.predict(test_data).flatten()
keras_dnn_err = test_labels.flatten() - dnn_out
plt.plot(keras_dnn_err)
plt.grid()
plt.title('DNN Error for random data')
plt.show()

dnn_fft_mse = 10*np.log10(np.mean(pow(np.abs(keras_dnn_err),2)))
labels_sigpow = 10*np.log10(np.mean(pow(np.abs(test_labels.flatten()),2)))
print("Neural Network SNR compare to SciPy FFT: ", labels_sigpow - dnn_fft_mse)

