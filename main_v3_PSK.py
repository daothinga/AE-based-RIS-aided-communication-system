# PSK for communications with Reconfigurable Intelligent Surfaces


# Import libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Layer, Dense
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import backend as K
import time 
from tensorflow.keras.layers import Lambda

# tf.config.run_functions_eagerly(True)


# Declare parameters
k = 3 # Total 2^k symbols
n_x = 2**k
n_val = 2**k * 6

# Prepare dataset
x_train = np.eye(n_x)
x_test = np.eye(n_x)



x_train = np.tile(x_train,(10**3,1))
x_test = np.tile(x_test,(10**4,1))
batch_size = 8

# inputSyms = np.random.randint(low=0, high = n_x, size=n_x*10**2) #Random 1's and 0's as input to BPSK modulator
inputSyms_train = np.tile(np.arange(n_x), (10**3))
inputSyms_test = np.tile(np.arange(n_x), (10**4))
# print()
constellation = np.zeros(shape=(n_x,2))
constellation[:,0] = np.cos(np.arange(0,n_x)/n_x*2*np.pi)  #reference constellation for M-PSK
constellation[:,1] = np.sin(np.arange(0,n_x)/n_x*2*np.pi) 

constellation_train = constellation[inputSyms_train]
print('Shape of constellation: ', np.shape(constellation_train))

# print(constellation_train)
constellation_test = constellation[inputSyms_test]

print(np.shape(x_train))

# x_train_label = np.argmax(x_train, axis=1)
# print(x_train_label)



############### Add noise layer with a fixed variance belta 
n = 2   # Channel use
R = k/n  # Communication rate (bit/channel use)
Eb_No_dB = 14
belta = 1/(2*R*(10**(Eb_No_dB/10)))    # Variance
belta_sqrt = np.sqrt(belta)            # Derivation
print('belta_sqrt: ', belta_sqrt)
n_samples = np.shape(x_train)[0]
n_x_test = np.shape(x_test)[0]
print('n_x_test: ', n_x_test)

model_name = "my_model_" + str(k) + '_' + str(Eb_No_dB)

N = 160  # The number of elements in RIS
# M = 3  # The number of codewords

### hri

################ Compute Bit error rate as loss function for model training
def BLER(y_true, y_pred):
  y_true_index = tf.argmax(y_true, axis=1)

  y_pred_index = tf.argmax(tf.math.round(y_pred), axis=1)
  BLER_ = K.mean(K.not_equal(y_true_index, y_pred_index))
    
  return BLER_


def BLER_numpy(y_true, y_pred):
  y_true_index = np.argmax(y_true, axis=1)
  y_pred_index = np.argmax(np.round(y_pred), axis=1)

  BLER_ = np.mean(np.not_equal(y_true_index, y_pred_index)) 
    
  return BLER_


class Channel_layer_TR(Layer):

  def __init__(self, initial_value):
      super(Channel_layer_TR, self).__init__()
      # Create a non-trainable weight.
      self.weight = tf.Variable(initial_value=initial_value, trainable=False, dtype=tf.float32)

  def call(self, inputs):    
    # print('Channel_layer_TR')
    # print(inputs)
    # print(inputs[:,0:1])
    # print(self.weight)
    return tf.concat([tf.matmul(inputs[:,0:1], self.weight), tf.matmul(inputs[:,1:2], self.weight)], axis=1)   # shape (None, 2N)


class Channel_layer_RI(Layer):
  def __init__(self, initial_value):
      super(Channel_layer_RI, self).__init__()
      # Create a non-trainable weight
      self.weight = tf.Variable(initial_value=initial_value, trainable=False, dtype=tf.float32)

  def call(self, inputs):    
    # print('Channel_layer_RI')
    # print(inputs)
    # print(self.weight)
    return tf.concat([tf.matmul(inputs[:,0:N], self.weight), tf.matmul(inputs[:,N:2*N], self.weight)], axis=1)   # shape (None, 2)


############### Encoder
Input = tf.keras.Input(shape=(2), dtype=tf.float64)
# model_encode = tf.keras.Sequential()
# model_encode.add(Dense(units=64, activation='relu'))
# model_encode.add(Dense(units = 2, activation='linear'))    # Shape: (None, 2)






###### Channel from transmitter to RIS
alpha = 0.065     # Degree of phase-shift, near 1 means less phase shift
# htr = np.random.rand(1,N)
htr = alpha*np.random.randint(low=20, high=100, size=(1,N))/100
print('htr: ', htr)
hri = alpha*np.random.randint(low=20, high=100, size=(N,1))/100

model_TR = tf.keras.Sequential() 
model_TR.add(Channel_layer_TR(initial_value=htr))



####### Reconfigurable Intelligent Surfaces (RIS)
hri_tile = tf.tile(tf.transpose(hri),tf.constant([batch_size,1]))
hri_tile = tf.cast(hri_tile, tf.float32)
# print('hri_tile: ', hri_tile)


# c_ should be a fix value for a given receiver's location
amplitude_attenuation = 0.5   # 3dB

a = np.ones(64) 
b = -np.ones(96)

c_ris = amplitude_attenuation*np.concatenate([a, b])
c_ris = np.tile(c_ris, [2])


###################  Channel from RIS to receiver
model_RI = tf.keras.Sequential()
model_RI.add(Channel_layer_RI(initial_value=hri))



# Add noise
# noise_Gaussian = tf.random.normal(shape=(batch_size,2), mean=0, stddev=belta_sqrt)
noise_Gaussian = tf.keras.Input(shape=(n))

model_decode = tf.keras.Sequential()
model_decode.add(Dense(64, use_bias = True, activation='relu'))
model_decode.add(Dense(units = n_x, use_bias = True, activation = 'softmax'))



######################  Training Step ################
#########################################################
# x_encode = model_encode(Input)
# x_encode = x_encode / tf.sqrt(tf.reduce_mean(tf.math.reduce_sum(tf.math.multiply(x_encode, x_encode),axis=1)))
# x_encode = x_encode / tf.sqrt(n*tf.reduce_mean(tf.math.multiply(x_encode, x_encode)))
# x_encode = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(x_encode) #

x_ris = model_TR(Input) 

y_ris = tf.multiply(x_ris, c_ris)       # Comment out this line to consider constant phase shift
y = model_RI(y_ris)

y_mean = tf.sqrt(tf.reduce_mean(tf.math.reduce_sum(tf.math.multiply(y, y),axis=1)))
y = tf.divide(y, y_mean) + noise_Gaussian

# y = y + noise_Gaussian
y_decode = model_decode(y)


com_model = tf.keras.Model(inputs=[Input, noise_Gaussian], outputs=y_decode)
loss_fn = tf.keras.losses.CategoricalCrossentropy()    
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
com_model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy', BLER])

n_rounds = 3
for i in range(n_rounds):
    print('Training Round: ', i)
    noise_train = np.random.normal(0, belta_sqrt, (np.shape(x_train)[0],n))
    noise_test = np.random.normal(0, belta_sqrt, (np.shape(x_test)[0],n))
   
    if i == (n_rounds-1):
      his = com_model.fit([constellation_train, noise_train], x_train, validation_data = ([constellation_test, noise_test], x_test), batch_size = batch_size, epochs=1, verbose=1)
    else:
      com_model.fit([constellation_train, noise_train], x_train, batch_size = batch_size, epochs=10, verbose=1)

results = his.history # Example: {'loss': [1.0246219635009766], 'accuracy': [0.7440000176429749], 'val_loss': [0.39520153403282166], 'val_accuracy': [0.9375]}
tmp = results['val_accuracy'][0]


BLER = []

for Eb_No_dB in np.arange(-2.0, 16.0, 0.5):
    belta = 1/(2*R*(10**(Eb_No_dB/10)))
    belta_sqrt = np.sqrt(belta)
    noise_try = belta_sqrt * np.random.randn(np.shape(x_test)[0],n)
    print(np.shape(noise_try))

    decoded_sys_round = np.round(com_model.predict([constellation_test,noise_try]))

    block_error_rate = BLER_numpy(x_test, decoded_sys_round)
    BLER.append(block_error_rate)
    print('Eb_No_dB: ', Eb_No_dB)
    print('Error Rate: ', block_error_rate)

np.savetxt('BLER_sim_8_PSK_RIS.txt', BLER)
# np.savetxt('BLER_sim_16_PSK_RIS.txt', BLER)

plt.yscale('log')
plt.plot(np.arange(-2.0, 16.0, 0.5),BLER,'r.-')
plt.grid(True)
plt.ylim(10**-5, 1)
plt.xlim(-2, 16)
plt.title("Block error rate")
plt.show()



# Save in a file

"""
### Visualization 
x_val = x_train[:n_val,:]
x_ = model_encode(x_val).numpy()
# x_encode_ = tf.math.l2_normalize(x_).numpy()
# x_encode_ = x_ / np.sqrt(np.mean(np.sum(x_**2, axis=1)))
x_encode_ = x_ / np.sqrt(n*np.mean(x_**2))
print('mean of x_encode: ', np.mean(np.sum(x_encode_**2, axis=1)))
noise = tf.random.normal(shape=(np.shape(x_val)[0],2), mean=0, stddev=belta_sqrt).numpy()



x_ris_ = model_TR(x_encode_)
hri_tile_ = np.tile(np.transpose(hri),(n_val,1))
input_ris_ = np.concatenate((x_ris_, hri_tile_), axis=1)


y_ris_ = np.multiply(x_ris_, c_ris)          # Comment out this line to consider constant phase shift
y_RxRe = model_RI(y_ris_)
# y_mean = np.sqrt(np.mean(np.sum(y_RxRe**2, axis=1)))
y_mean = np.sqrt(np.mean(y_RxRe**2))
y_RxRe = y_RxRe/y_mean
y_noise_ = y_RxRe + noise


# plt.plot(y_noise_[:,0], y_noise_[:,1], 'k.', markersize = 8)
plt.plot(x_encode_[:,0], x_encode_[:,1], 'k.', markersize=15)
# plt.plot(y_RxRe[:,0], y_RxRe[:,1], 'gx', markersize=12)
plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5', '0', '0.5', '1.0'])
plt.yticks([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5', '0', '0.5', '1.0'])
fig_name = 'fig_' + str(k) + '.eps'
# plt.figure(figsize=(6,6))

plt.savefig(fig_name)
plt.show()
"""





      
