# -*- coding: utf-8 -*-
"""
Lib_Wavenets_01.py
Wavenet Library 
"""

from __future__ import absolute_import
from __future__ import print_function


import numpy as np; #np.set_printoptions(threshold=np.nan)

from keras.utils import np_utils
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation,  Flatten, Reshape, Permute,  Lambda, Input
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, ZeroPadding1D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import ConvLSTM2D, LSTM, CuDNNLSTM, RepeatVector,BatchNormalization
from keras.layers import concatenate, add, multiply

from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.regularizers import l2
from keras.initializers import TruncatedNormal

from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import Adam

from keras import backend as K           #K.image_data_format()
K.set_image_data_format('channels_last') #K.image_data_format()


adam = Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#init_=TruncatedNormal(mean=0.0, stddev=0.05,seed=42) ||glorot_normal||glorot_uniform
#kernel_regularizer=l2(0.001) 

def WaveNetBlock(nb_filter, kernel_size, dilation_rate, n_conv_filters=1, conv_filter_size=1, type_=0):
    def f(input_):
        '''WaveNetBlocks library
        INPUT:
            filters
            kernel_size
            dilation_rate
            conv_filter_size
            type_ (Inner_padding, OutPadding, use_bias, graph_config): 0 ('same', 'same'), 1 ('casual', 'casual'), 
        OUTPUT: 
            '''
        params =	{0: ['same',   'same',     True,   0],
                  1: ['causal', 'causal',   False,  0],
                  2: ['same',   'same',     True,   1],
                  3: ['causal', 'causal',   False,  1],
                  4: ['same',   'same',     True,   2],
                  5: ['causal', 'causal',   False,  2],
                  6: ['same',   'same',     True,   3],
                  7: ['causal', 'causal',   False,  3],
                  8: ['same',   'same',     True,   4],
                  9: ['causal', 'causal',   False,  4],
                  10:['same',   'same',     True,   5],
                  11:['causal', 'causal',   False,  5]}

        #lecun_uniform || lecun_normal ||glorot_normal ||glorot_uniform ||he_normal ||he_uniform
        init_='he_normal' 
        
        residual =input_
        if params[type_][3]<=1:
            layer_out =   Conv1D(nb_filter, kernel_size,
                                 dilation_rate=dilation_rate,
                                 activation='linear', padding=params[type_][0], 
                                 use_bias=params[type_][2],
                                 kernel_initializer=init_)(input_)
            layer_out =   Activation('selu')(layer_out)
            skip_out =    Conv1D(n_conv_filters,conv_filter_size, 
                                 activation='linear', 
                                 use_bias=params[type_][2],
                                 kernel_initializer=init_)(layer_out)
            if params[type_][3]==0: #type_=1
                network_out =add([residual, skip_out])
            else:  #type_=1
                network_in =  Conv1D(n_conv_filters, conv_filter_size,  activation='linear',#'relu'
                                     use_bias=params[type_][2],
                                     kernel_initializer=init_)(layer_out)
                network_out =add([residual, network_in])                
        
        else:      #type_=2
            tanh_out   =Conv1D(nb_filter, kernel_size, 
                               dilation_rate=dilation_rate,
                               activation='tanh', padding=params[type_][0],
                               use_bias=params[type_][2],
                               kernel_initializer=init_)(input_)
            sigmoid_out=Conv1D(nb_filter, kernel_size,
                               dilation_rate=dilation_rate,
                               activation='sigmoid',padding=params[type_][0],
                               use_bias=params[type_][2],
                               kernel_initializer=init_)(input_)
            merged = multiply([tanh_out, sigmoid_out])
            skip_out = Conv1D(n_conv_filters,n_conv_filters, activation='relu',
                              padding=params[type_][0])(merged)
            network_out = add([skip_out, residual])
        
        return network_out, skip_out
    return f 



def WaveNet_ts(in_shape=(72,3), out_shape=(3,), nb_filter_=32, kernel_size_=2, flag_type=0):
    '''Time_series wavenet
    shape_in=(times,channels)'''
    input_= Input(shape=in_shape)
    
    l1a, l1b = WaveNetBlock(nb_filter_, kernel_size_, 1,  type_=flag_type)(input_) 
    l2a, l2b = WaveNetBlock(nb_filter_, kernel_size_, 2,  type_=flag_type)(l1a)
    l3a, l3b = WaveNetBlock(nb_filter_, kernel_size_, 4,  type_=flag_type)(l2a)
    l4a, l4b = WaveNetBlock(nb_filter_, kernel_size_, 8,  type_=flag_type)(l3a)
    l5a, l5b = WaveNetBlock(nb_filter_, kernel_size_, 16, type_=flag_type)(l4a)
    l6a, l6b = WaveNetBlock(nb_filter_, kernel_size_, 32, type_=flag_type)(l5a) #l7a/b in the paper
    
    l5b = Dropout(0.2)(l5b) #0.8 in the paper
    l6b = Dropout(0.2)(l6b) #0.8 in the paper
    
    l7 =    add([l1b, l2b, l3b, l4b, l5b, l6b])
    #l7 =    concatenate([l1b, l2b, l3b, l4b, l5b, l6b])
    l8 =    Activation('relu')(l7)
    l9 =    Conv1D(32,1, activation='linear', use_bias=False,
                   kernel_initializer='he_normal')(l8) #,kernel_regularizer=l2(0.001)
    out= Conv1D(1, 1, activation='linear', use_bias=False)(l9) 
    out= Flatten()(out)
    out = Dropout(0.2)(out) 
    out= Dense(out_shape[0], activation='linear')(out)
    out=  CuDNNLSTM(out_shape[0], return_sequences=False)(out)
    model = Model(inputs=input_, outputs=out)
    return model

def WaveNet_ts2(in_shape=(72,1), out_shape=(1,), nb_filter_=32, kernel_size_=2, flag_type=0):
    '''Time_series wavenet
    shape_in=(times,channels)'''
    input_= Input(shape=in_shape)
    
    l1a, l1b = WaveNetBlock(nb_filter_, kernel_size_, 1,  type_=flag_type)(input_) 
    l2a, l2b = WaveNetBlock(nb_filter_, kernel_size_, 2,  type_=flag_type)(l1a)
    l3a, l3b = WaveNetBlock(nb_filter_, kernel_size_, 4,  type_=flag_type)(l2a)
    l4a, l4b = WaveNetBlock(nb_filter_, kernel_size_, 8,  type_=flag_type)(l3a)
    l5a, l5b = WaveNetBlock(nb_filter_, kernel_size_, 16,  type_=flag_type)(l4a)
    l6a, l6b = WaveNetBlock(nb_filter_, kernel_size_, 32,  type_=flag_type)(l5a) #l7a/b in the paper
    
    l5b = Dropout(0.2)(l5b) #0.8 in the paper
    l6b = Dropout(0.2)(l6b) #0.8 in the paper
    
    #l7 =    add([l1b, l2b, l3b, l4b, l5b, l6b])
    l7 =    concatenate([l1b, l2b, l3b, l4b, l5b, l6b])
    l8 =    Activation('relu')(l7)
    #l9 =    Conv1D(32,1, activation='linear', use_bias=False,
    #               kernel_initializer='he_normal')(l8) #,kernel_regularizer=l2(0.001)
    out= Flatten()(l8)
    out= Activation('relu')(out)
    out= Dropout(0.2)(out) 
    out= Dense(out_shape[0], activation='linear')(out)
    #out=  CuDNNLSTM(out_shape[0], return_sequences=False)(out)
    model = Model(inputs=input_, outputs=out)
    return model


'''Attention  Backward'''
'''
for ftype in range(0,6):
    for ks in range(2,6):
        with open('C:\\Programming\\Python\\NS_libs\\fmain_vars.pickle', 'rb') as f:
            xX_main1, yY_main1, xX_test1, yY_test1, stats_loads1= pickle.load(f)
            xX_main=xX_main1; yY_main=yY_main1; xX_test=xX_test1; yY_test=yY_test1; stats_loads=stats_loads1;
        xX_main=xX_main[0]; 
        xX_test=xX_test[0]; 
        xX_main=xX_main[:,:,38:]      #xX_main=np.delete(xX_main, [32,33], axis=2)
        xX_test=xX_test[:,:,38:]      #xX_test=np.delete(xX_test, [32,33], axis=2)
        x_mainCal=xX_main1[0]; x_mainCal=x_mainCal[:,:, 1:9] 
        x_testCal=xX_test1[0]; x_testCal=x_testCal[:,:, 1:9]
        xX_main_c=np.concatenate((xX_main,x_mainCal), axis=2) #xX_main #
        xX_test_c=np.concatenate((xX_test,x_testCal), axis=2) #xX_test #
        
        
        y_mainCal=xX_main1[1]; y_mainCal=y_mainCal[:,3:7, [1,2,3,4,5,6,7,8,34]] #Season & HOUR
        y_testCal=xX_test1[1]; y_testCal=y_testCal[:,3:7, [1,2,3,4,5,6,7,8,34]] #Season & HOUR
        
        yY_main=yY_main[:,5,0:1]
        yY_test=yY_test[:,5,0:1] 
        
        file_best_model_weights="C:\\Programming\\Python\\NS_libs\\weights.hdf5"
        callbacks_ = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=0), #reduce_lr, #clr,
                     ModelCheckpoint(filepath=file_best_model_weights, monitor='val_loss', save_best_only=True, verbose=0)]
        #opt=Adam(lr=.001, decay=.000001)  #SGD(lr=0.0001, decay=.000001)
        opt=adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
        modelwv= WaveNet_ts2(in_shape=(72,9), out_shape=(1,), nb_filter_=32, kernel_size_=ks, flag_type=ftype)
        #modelwv=build_D1A1_X1recnn2(in_shape=[(24,4),(24,3)], out_shape=(24,1), nb_filters=32, filter_length=5, drop_out=0.25, 
        #                 out_layer=["linear", "softmax"], conv_type=1, attn_type=3, maxpool_=1, type=0):
        modelwv.compile(optimizer=opt, loss='mse')
        history=modelwv.fit(xX_main_c[srow:,:,:],yY_main[srow:,:],
                            batch_size=25000, epochs=5000,
                            validation_data=(xX_test_c[:,:,:],yY_test),
                            callbacks=callbacks_)
        
        modelwv.load_weights(file_best_model_weights)
        K.set_value(modelwv.optimizer.lr, 0.0000001)
        history=modelwv.fit(xX_main_c[srow:,:,:],yY_main[srow:,:],
                            batch_size=25000, epochs=3000,
                            validation_data=(xX_test_c[:,:,:],yY_test),
                            callbacks=callbacks_)
        
        plot_history(history) 
        modelwv.load_weights(file_best_model_weights)
        rdY_pred=modelwv.predict(xX_test_c)
        rdY_predRavel=rdY_pred.ravel()
        yY_testRavel=yY_test.ravel()
        adY_pred=       de_normalize(rdY_predRavel, stats_loads[1][0], stats_loads[2][0], if_log=bool(stats_loads[0]))
        adY_test=  de_normalize(yY_testRavel, stats_loads[1][0], stats_loads[2][0], if_log=bool(stats_loads[0]))
        astatsTbl=loss_table(predictions=adY_pred, targets=adY_test,  CaseName_='WaveNet_II_type_'+str(ftype)+'_ks_'+str(ks),
                            table=astatsTbl, ax_=None, ltype='all', printFlag_=False)

'''
















'''Attention  Backward'''
'''
with open('C:\\Programming\\Python\\NS_libs\\fmain_vars.pickle', 'rb') as f:
    xX_main1, yY_main1, xX_test1, yY_test1, stats_loads1= pickle.load(f)
    xX_main=xX_main1; yY_main=yY_main1; xX_test=xX_test1; yY_test=yY_test1; stats_loads=stats_loads1;

xX_main=xX_main[0]; 
xX_test=xX_test[0]; 
xX_main=xX_main[:,:,38:]      #xX_main=np.delete(xX_main, [32,33], axis=2)
xX_test=xX_test[:,:,38:]      #xX_test=np.delete(xX_test, [32,33], axis=2)
x_mainCal=xX_main1[0]; x_mainCal=x_mainCal[:,:, 1:9] 
x_testCal=xX_test1[0]; x_testCal=x_testCal[:,:, 1:9]
xX_main_c=np.concatenate((xX_main,x_mainCal), axis=2) #xX_main #
xX_test_c=np.concatenate((xX_test,x_testCal), axis=2) #xX_test #


y_mainCal=xX_main1[1]; y_mainCal=y_mainCal[:,3:7, [1,2,3,4,5,6,7,8,34]] #Season & HOUR
y_testCal=xX_test1[1]; y_testCal=y_testCal[:,3:7, [1,2,3,4,5,6,7,8,34]] #Season & HOUR

yY_main=yY_main[:,5,0:1]
yY_test=yY_test[:,5,0:1] 

file_best_model_weights="C:\\Programming\\Python\\NS_libs\\weights.hdf5"
callbacks_ = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=0), #reduce_lr, #clr,
             ModelCheckpoint(filepath=file_best_model_weights, monitor='val_loss', save_best_only=True, verbose=0)]
#opt=Adam(lr=.001, decay=.000001)  #SGD(lr=0.0001, decay=.000001)
opt=adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
modelwv= WaveNet_ts3(in_shape=(72,9), out_shape=(1,), nb_filter_=32, kernel_size_=3, flag_type=5)


#modelwv=build_D1A1_X1recnn2(in_shape=[(24,4),(24,3)], out_shape=(24,1), nb_filters=32, filter_length=5, drop_out=0.25, 
#                 out_layer=["linear", "softmax"], conv_type=1, attn_type=3, maxpool_=1, type=0):

modelwv.compile(optimizer=opt, loss='mse')
history=modelwv.fit(xX_main_c[srow:,:,:],yY_main[srow:,:],
                    batch_size=25000, epochs=5000,
                    validation_data=(xX_test_c[:,:,:],yY_test),
                    callbacks=callbacks_)

modelwv.load_weights(file_best_model_weights)
K.set_value(modelwv.optimizer.lr, 0.0000001)
history=modelwv.fit(xX_main_c[srow:,:,:],yY_main[srow:,:],
                    batch_size=25000, epochs=3000,
                    validation_data=(xX_test_c[:,:,:],yY_test),
                    callbacks=callbacks_)

plot_history(history) 
modelwv.load_weights(file_best_model_weights)
rdY_pred=modelwv.predict(xX_test_c)
rdY_predRavel=rdY_pred.ravel()
yY_testRavel=yY_test.ravel()
adY_pred=       de_normalize(rdY_predRavel, stats_loads[1][0], stats_loads[2][0], if_log=bool(stats_loads[0]))
adY_test=  de_normalize(yY_testRavel, stats_loads[1][0], stats_loads[2][0], if_log=bool(stats_loads[0]))
astatsTbl=loss_table(predictions=adY_pred, targets=adY_test,  CaseName_='WaveNet_II_type_5_kernel_3',
                    table=astatsTbl, ax_=None, ltype='all', printFlag_=False)






def AdSeqBlock(nb_filter, kernel_size, dilation_rate, n_conv_filters=1, conv_filter_size=1, type_=0, flag_wave=1, name='ASeqB_1'):
    def f(input_):
        features_Branch1, fB1size =CNNRNN_MIMO_v01.build_feature_seqs(inputs=inputs1, nb_filter=nb_filters, 
                                                             filter_length=1, 
                                                             drop_out=drop_out, 
                                                             activation_=out_layer[0],
                                                             conv_type=conv_type, 
                                                             maxpool_=maxpool_, name_='AS') 
        features_Branch3, fB3size =CNNRNN_MIMO_v01.build_feature_seqs(inputs=inputs1, nb_filter=nb_filters, 
                                                             filter_length=3, 
                                                             drop_out=drop_out, 
                                                             activation_=out_layer[0],
                                                             conv_type=conv_type, 
                                                             maxpool_=maxpool_) 
        features_Branch5, fB5size =CNNRNN_MIMO_v01.build_feature_seqs(inputs=inputs1, nb_filter=nb_filters, 
                                                             filter_length=5, 
                                                             drop_out=drop_out, 
                                                             activation_=out_layer[0],
                                                             conv_type=conv_type, 
                                                             maxpool_=maxpool_) 
        features_Branch7, fB7size =CNNRNN_MIMO_v01.build_feature_seqs(inputs=inputs1, nb_filter=nb_filters, 
                                                             filter_length=7, 
                                                             drop_out=drop_out, 
                                                             activation_=out_layer[0],
                                                             conv_type=conv_type, 
                                                             maxpool_=maxpool_)
        if flag_wave==1:
            nb_filter_=32
            kernel_size_=3
            flag_type=5
            l1a, l1b = WaveNetBlock(nb_filter_, kernel_size_, 1)(inputs1) 
            l2a, l2b = WaveNetBlock(nb_filter_, kernel_size_, 2)(l1a)
            l3a, l3b = WaveNetBlock(nb_filter_, kernel_size_, 4)(l2a)
            l4a, l4b = WaveNetBlock(nb_filter_, kernel_size_, 8)(l3a)
            l5a, l5b = WaveNetBlock(nb_filter_, kernel_size_, 16)(l4a)
            l6a, l6b = WaveNetBlock(nb_filter_, kernel_size_, 32)(l5a) #l7a/b in the paper
            l5b = Dropout(0.2)(l5b) #0.8 in the paper
            l6b = Dropout(0.2)(l6b) #0.8 in the paper
            l7 =    concatenate([l1b, l2b, l3b, l4b, l5b, l6b])
            l8 =    Activation('relu')(l7)
            resLayer=concatenate([features_Branch1, features_Branch3, features_Branch5, features_Branch7, l8], axis=-1, name='AAA') #axis=-1
            resLayer_size=(fB1size[0], fB1size[1]+ fB3size[1]+fB5size[1]+fB7size[1]+6)
        else:
            resLayer=concatenate([features_Branch1, features_Branch3, features_Branch5, features_Branch7], axis=-1, name='AAA') #axis=-1
            resLayer_size=(fB1size[0], fB1size[1]+ fB3size[1]+fB5size[1]+fB7size[1])

      return network_out, skip_out
return f 
'''















