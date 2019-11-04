'''Lib_NN_01.py library
BASIC UTILS AND MODELS
'''

from __future__ import absolute_import
from __future__ import print_function


import glob, datetime, time, os, sys
import numpy as np
#np.set_printoptions(threshold=np.nan)
import pandas as pd

from sklearn import linear_model

from keras import backend as K
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.models import Model
from keras.models import Sequential, Model
from keras.models import model_from_json

from keras.layers.merge import Concatenate, add
from keras.layers import Input, merge, Flatten, Dense, Activation, Convolution1D, ZeroPadding1D,TimeDistributed
from keras.layers import TimeDistributed, Reshape, Flatten, BatchNormalization
from keras.layers.recurrent import LSTM

from keras.layers.core import Dense, Dropout, Activation,  Flatten, Reshape, Permute,  Lambda, RepeatVector
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, UpSampling1D, UpSampling2D, ZeroPadding1D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

import matplotlib.pyplot as plt
from keras.optimizers import Adam, Adagrad,Adadelta,Adamax,Nadam,SGD,RMSprop

path = "./training_data_large/"  # to make sure signal files are written in same directory as data files

#seed=777
#from numpy.random import seed; seed(777)
#from tensorflow import set_random_seed;
#set_random_seed(777)


def draw_model(model, filename=''):
    '''Plot the model and save the graph with dims to file
    EX:
        draw_model(model3, filename='C:\work\model3me.png')
        SVG(model_to_dot(model3).create(prog='dot', format='svg'))
        '''
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    from keras.utils.vis_utils import plot_model
    if filename=='':
        filename=model.name+'.png'
    plot_model(model, to_file=filename, show_shapes=True)


def save_neuralnet (model, model_name, locpath=None):
    '''writing a model with the name model_name to folder locpath'''
    if locpath==None:
        locpath="./"
    json_string = model.to_json()
    open(locpath + model_name + '_architecture.json', 'w').write(json_string)
    model.save_weights(locpath + model_name + '_weights.h5', overwrite=True)
    yaml_string = model.to_yaml()
    with open(locpath + model_name + '_data.yml', 'w') as outfile:
        outfile.write( yaml_string)

def load_neuralnet(model_name, locpath=None):
    """ 
    Reading the model from disk - including all the trained weights and the complete model design (hyperparams, planes,..)
    """
    if locpath==None:
        locpath="./"
    arch_name = locpath + model_name + '_architecture.json'
    weight_name = locpath + model_name + '_weights.h5'
    if not os.path.isfile(arch_name) or not os.path.isfile(weight_name):
        print("model_name given and file %s and/or %s not existing. Aborting." % (arch_name, weight_name))
        sys.exit()
    print("Loaded model: ",model_name)
    try:
        model = model_from_json(open(arch_name).read(),{'Convolution1D_Transpose_Arbitrary':Convolution1D_Transpose_Arbitrary})
    except NameError: # are we on Theano?
        model = model_from_json(open(arch_name).read())

    model.load_weights(weight_name)
    return model

def plot_history(history):
    '''Plot training 
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    '''
    # "Loss"
    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'][2:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def shape_tuple(input):
    '''Return a tuple of shape for (numpy) Input, w/o first obs dimention
    return tuple of remaining dimentions'''
    shape= input.shape
    input_shape=shape[1:]
    return input_shape

def basic_dense(Input1_data,
                Output1_shape=(1,),
                init="glorot_uniform",
                num_neurols=512,
                dropout_r=0.5,
                rrd=.75,
                mname=None): #shape and rate reduction 
    '''Basic dense model
    Input1 {num_obs X num_times X num_features}
    init="lecun_uniform",
    Output1_shape->output shape, EXCLUDES num_obs dimention
    Sample: model=basic_dense(X_train, Output1_shape=(24,15))
    '''
    flat_shape=1
    for i in range(0,len(Output1_shape)):
            flat_shape=flat_shape*Output1_shape[i]
            
    input_shape=shape_tuple(Input1_data)
    model = Sequential()
    model.add(Dense(num_neurols, input_shape=input_shape, kernel_initializer=init, activation="relu"))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_r))
    model.add(Flatten())
    model.add(Dense(np.int64(num_neurols*rrd), kernel_initializer=init, activation="relu"))
    model.add(Dropout(rate=dropout_r*rrd))
    model.add(BatchNormalization())
    model.add(Dense(flat_shape, kernel_initializer=init, activation="linear"))
    model.add(Reshape(target_shape=Output1_shape))
    if mname!=None:
        model.name=mname
    return model

def basic_denseTemp(input_shape=1849,
                output_shape=1,
                init="glorot_uniform",
                nb_filter=1024,
                dropout_r=.2,
                rrd=.6,
                mname=None): #shape and rate reduction
    '''Temp Basic dense model
    
    ''' 
    model = Sequential()
    model.add(Dense(nb_filter, input_dim=input_shape, kernel_initializer=init, activation="relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_r))
    model.add(Dense(np.int64(nb_filter*rrd), kernel_initializer=init, activation="relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_r*rrd))
    model.add(Dense(output_shape, kernel_initializer=init, activation="linear"))
    if mname!=None:
        model.name=mname
    return model 


def basic_denseTemp2(input_shape=1849,
                output_shape=1,
                init="glorot_uniform",
                nb_filter=1024,
                dropout_r=.2,
                rrd=.6,
                mname=None): #shape and rate reduction
    '''Temp Basic dense model
    
    ''' 
    model = Sequential()
    model.add(Dense(nb_filter, input_dim=input_shape, kernel_initializer=init, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_r))
    model.add(Dense(np.int64(nb_filter*rrd*rrd), kernel_initializer=init, activation="linear"))
    model.add(Dense(np.int64(nb_filter*rrd*rrd), kernel_initializer=init, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_r*rrd*rrd*rrd))
    model.add(Dense(output_shape, kernel_initializer=init, activation="linear"))
    if mname!=None:
        model.name=mname
    return model 


def basic_recurrent_1(input_lag=48,
               nb_input_ts=11, 
               out_fcst=24, 
               nb_out_ts=15,  
               nb_filter=512,
               filter_length=3,
               class_mode=None,
               activation="relu",
               init="glorot_uniform",#lecun_uniform / glorot_uniform
               mname=None,
               recur_type='CuDNNGRU'):  #LSTM/ GRU/ CuDNNGRU/ CuDNNLSTM/CuDNNGRU
    """:Return: LSTM, GRU, CU Model for predicting the next values in a timeseries given a fixed-size lookback window of previous values.
    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
    :param int input_lag:    lag (lookback) of input time-series
    :param int nb_input_ts:  number of input timeseries 
    :param int nb_out_ts:    number of output timeseries, often equal to the number of inputs.
    :param int out_fcst:     number of time periods to forecast, for each timeseries
    """
    model = Sequential()
    if recur_type=='GRU':
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=True, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    elif recur_type=='CuDNNGRU':
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=True, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    elif recur_type=='CuDNNLSTM':
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=True, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=True, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    
    
    model.add(TimeDistributed(Dense(nb_filter)))
    model.add(BatchNormalization())
    '''
    if recur_type=="GRU":
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=True))
    elif recur_type=="CuDNNGRU":
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=True))
    elif recur_type=="CuDNNLSTM":
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=True))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=True))
    '''
    
    #model.add(BatchNormalization())
    #model.add(TimeDistributed(Dense(nb_filter)))
    
    if recur_type=="GRU":
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=False))
    elif recur_type=="CuDNNGRU":
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=False))
    elif recur_type=="CuDNNLSTM":
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=False))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=False))
    
    model.add(RepeatVector(out_fcst))
    model.add(Dense(nb_out_ts,  activation='linear')) # For binary classification, change the activation to 'sigmoid'
    if mname!=None:
        model.name=mname
    return model


def recurrent_equisize(input_lag=48,
               nb_input_ts=11, 
               out_fcst=24, 
               nb_out_ts=15,  
               nb_filter=512,
               filter_length=3,
               class_mode=None,
               activation="relu",
               init="glorot_uniform",#lecun_uniform / glorot_uniform
               mname=None,
               recur_type='CuDNNGRU'):  #LSTM/ GRU/ CuDNNGRU/ CuDNNLSTM/CuDNNGRU
    """:Return: LSTM, GRU, CU Model for predicting the next values in a timeseries given a fixed-size lookback window of previous values.
    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
    :param int input_lag:    lag (lookback) of input time-series
    :param int nb_input_ts:  number of input timeseries 
    :param int nb_out_ts:    number of output timeseries, often equal to the number of inputs.
    :param int out_fcst:     number of time periods to forecast, for each timeseries
    """
    model = Sequential()
    if recur_type=='GRU':
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=False, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    elif recur_type=='CuDNNGRU':
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=False, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    elif recur_type=='CuDNNLSTM':
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=False, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=False, kernel_initializer=init,input_shape=(input_lag, nb_input_ts)))
    
    '''
    model.add(TimeDistributed(Dense(nb_filter)))
    model.add(BatchNormalization())
    
    if recur_type=="GRU":
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=True))
    elif recur_type=="CuDNNGRU":
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=True))
    elif recur_type=="CuDNNLSTM":
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=True))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=True))
    
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(nb_filter)))
    
    if recur_type=="GRU":
        model.add(GRU(units=nb_filter, stateful=False, return_sequences=False))
    elif recur_type=="CuDNNGRU":
        model.add(CuDNNGRU(units=nb_filter, stateful=False, return_sequences=False))
    elif recur_type=="CuDNNLSTM":
        model.add(CuDNNLSTM(units=nb_filter, stateful=False, return_sequences=False))
    else: #LSTM
        model.add(LSTM(units=nb_filter, stateful=False, return_sequences=False))
    '''
    model.add(RepeatVector(out_fcst))
    model.add(Dense(nb_out_ts,  activation='linear')) # For binary classification, change the activation to 'sigmoid'
    if mname!=None:
        model.name=mname
    return model

''' 
input = Input(shape=input_shape)
base_cnn_model = InceptionV3(include_top=False, ..)
temporal_analysis = TimeDistributed(base_cnn_model)(input)
conv3d_analysis = Conv3D(nb_of_filters, 3, 3, 3)(temporal_analysis)
conv3d_analysis = Conv3D(nb_of_filters, 3, 3, 3)(conv3d_analysis)
output = Flatten()(conv3d_analysis)
output = Dense(nb_of_classes, activation="softmax")(output)
from keras.layers.convolutional_recurrent import ConvLSTM2D 
#To use this layer, the video data has to be formatted as follows:
[nb_samples, nb_frames, width, height, channels] # if using dim_ordering = 'tf'
[nb_samples, nb_frames, channels, width, height] # if using dim_ordering = 'th'
    

callbacks = [
        ReduceLROnPlateau(patience=early_stopping_patience / 2, cooldown=early_stopping_patience / 4, verbose=1),
        EarlyStopping(patience=early_stopping_patience, verbose=1),]
    if not debug:
        callbacks.extend([
            ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5'),
                            save_best_only=True),
            CSVLogger(os.path.join(run_dir, 'history.csv')),
        ])

    if not debug:
        os.mkdir(checkpoint_dir)
        _log.info('Starting Training...')

   
t1_caldr_valid
data_dim = 10
timesteps = 4
batch_size = 32
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size, timesteps, data_dim), return_sequences=True, stateful=True))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch=50, batch_size=batch_size, validation_split=0.5)

# V1 **********
batch_size=20
output_dim=24
main_input = Input(shape=(96, 47), dtype='float32', name='main_input')
x=LSTM(output_dim=32, stateful=False, return_sequences=True)(main_input) #batch_input_shape=(None, 96, 47)
x=TimeDistributed(Dense(100, activation='relu'))(x)
x=LSTM(output_dim=64, stateful=False, return_sequences=False)(x) 
x=Dense(128, activation='relu')(x)
outputs=Dense(24, activation='linear')(x)
model1= Model(input=main_input, output=outputs)
model1.compile(optimizer='Adam',loss='mse')

# V2 **********
batch_size=20
output_dim=24
main_input = Input(shape=(96, 3), dtype='float32', name='main_input')
x=LSTM(output_dim=32, stateful=False, return_sequences=True)(main_input) #batch_input_shape=(None, 96, 47)
x=TimeDistributed(Dense(100, activation='relu'))(x)
x=LSTM(output_dim=64, stateful=False, return_sequences=False)(x) 
lstm_out=Dense(128, activation='relu')(x)

auxiliary_input = Input(shape=(96,44), name='aux_input')
x0=Flatten()(auxiliary_input)
#x0=TimeDistributed(Dense(20, activation='relu'))(auxiliary_input)

x = merge([lstm_out, x0], mode='concat')
x=LSTM(output_dim=64, stateful=False, return_sequences=False)(x) 
outputs=Dense(24, activation='linear')(x)
model2= Model(input=[main_input, auxiliary_input], output=outputs)
model2.compile(optimizer='Adam',loss='mse')

# V3 **********
model = Sequential()
model.add(LSTM(32, stateful=False, return_sequences=True, batch_input_shape=(None, 96, 47)))
model.add(TimeDistributed(Dense(100)))
model.add(LSTM(64, stateful=False, return_sequences=False))
model.add(Dense(128, activation='relu'))
model.add(Dense(24,activation='linear'))
model.compile(optimizer=Adam(), loss='mse')
model.summary()

#V4  ********


print('Creating Model')
model = Sequential()
model.add(LSTM(hidden_size, batch_input_shape=(batch_size, tsteps, inputs.shape[2]),
               return_sequences=True,
               stateful=True))
for i in range(2,lstm_layers_num):
    model.add(LSTM(hidden_size, batch_input_shape=(batch_size, tsteps, hidden_size),
                   return_sequences=True,
                   stateful=True))
model.add(LSTM(hidden_size,
               batch_input_shape=(batch_size, tsteps, hidden_size),
               return_sequences=False,
               stateful=True))
model.add(Dense(expected_output.shape[1]))
model.compile(loss='mse', optimizer='rmsprop')
print(model.summary())

bn_model=Sequential(get_bn_layers(p))
bn_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, val_labels))

bn_model.optimizer.lr = 1e-4
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=8, 
             validation_data=(conv_val_feat, val_labels))

bn_model.save_weights(path+'models/conv_512_6.h5')
bn_model.evaluate(conv_val_feat, val_labels)
bn_model.load_weights(path+'models/conv_512_6.h5')


'''













