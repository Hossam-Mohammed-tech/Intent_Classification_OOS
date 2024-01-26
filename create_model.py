# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:48:07 2023

@author: hossa
"""

from keras import Model
from keras.layers.core import Dense
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from tensorflow import keras
from tensorflow.keras import layers
import keras

def model_create(number_of_classes):
    ### Constructing the siamese model
    input_size_tsdae = 768
    input_size_use = 512
    
    # define two sets of inputs
    input_use = Input(shape=(input_size_use,))
    input_tsdae = Input(shape=(input_size_tsdae,))
    
    ##############
    ## the first branch operates on the first input
    x = Dense(512, activation='relu')(input_use)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    
    x = Model(inputs=input_use, outputs=x)
    
    ## the second branch opreates on the second input
    y = Dense(512, activation='relu')(input_tsdae)
    y = Dropout(0.4)(y)
    y = BatchNormalization()(y)
    y = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(y)
    y = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(y)
    y = Dropout(0.4)(y)
    y = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(y)
    y = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(y)
    
    y = Model(inputs=input_tsdae, outputs=y)
    
    ## combine the output of the two branches
    combined = concatenate([x.output, y.output])
    combined = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(combined)
    ###############
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    dense_layer = Dense(number_of_classes, activation='softmax', name='dense_layer')(combined)
    #norm_layer = Lambda(lambda  combined: K.l2_normalize(combined, axis=1), name='norm_layer')(dense_layer)
    
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=dense_layer)
    
    model.summary()

    return model