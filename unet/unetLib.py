from warnings import filters
import numpy as np
import tensorflow as tf
from tensorflow import keras

print('tensorflow:',tf.__version__)
print('tf,keras:',tf.keras.__version__)

from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Dropout, Flatten
from tensorflow.keras.layers import Activation, UpSampling2D, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import cv2 
import glob
import tarfile 
import numpy as np
import os

from skimage.io import imread, imshow
import eval_index as ev 



def conv2D_block(inputs=None, n_filters=64, maxpooling=True):
  """
    エンコーダレイヤー
    畳み込み層3つとマックスプーリング層1つのセット
    inputs= 入力画像 or nextLayer
    filters -> チャンネル数（1層目〜5層目:64, 128, 256, 512, 1024）
  """


  conv = Conv2D(n_filters, 
          kernel_size = (3, 3),  
          activation='relu', 
          padding = 'same'
        )(inputs)
  conv = Conv2D(n_filters,
          kernel_size = (3, 3), 
          activation=None, 
          padding = 'same'
        )(conv)
  # スキップ接続変数
  skip_connection = conv
  conv = BatchNormalization(axis=3)(conv)
  conv = Activation('relu')(conv)

  if maxpooling:
    next_layer = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv)
  else:
    next_layer = conv
  
  # conv2D_block[0] = next_layer, conv2D_block[1] = skip_connction
  return next_layer, skip_connection



def upsampling_block(ex_inputs, skip_connection, n_filters=64):
  """
    デコーダ
    アップサンプリングブロック
  """
  upsample = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=(2, 2),padding='same')(ex_inputs)
  # 同じ階層のエンコーダで生成された特徴マップと結合
  """
  concatenateとConcatenateがある。注意!
  """
  merge = concatenate([upsample, skip_connection], axis=3)
  conv = Conv2D(n_filters, 
          kernel_size = (3, 3), 
          padding='same', 
          activation='relu'
        )(merge)
  conv = Conv2D(n_filters, 
          kernel_size = (3, 3), 
          padding='same', 
          activation=None
        )(conv)
  conv = BatchNormalization(axis=3)(conv)
  conv = Activation('relu')(conv)

  return conv



def build_model(input_size=(256, 256, 3), n_filters=64, n_classes=1):
  """
    function to Build U-Net Model 
  """

  inputs = Input(input_size)

  # エンコーダ生成
  cBlock1 = conv2D_block(inputs, n_filters)
  cBlock2 = conv2D_block(cBlock1[0], 2*n_filters)
  cBlock3 = conv2D_block(cBlock2[0], 4*n_filters)
  cBlock4 = conv2D_block(cBlock3[0], 8*n_filters)
  cBlock5 = conv2D_block(cBlock4[0], 16*n_filters, maxpooling=False)
  print(type(cBlock5[0]))
  # デコーダ生成
  print(filter)
  uBlock1 = upsampling_block(cBlock5[0], cBlock4[1], 8*n_filters)
  uBlock2 = upsampling_block(uBlock1, cBlock3[1], 4*n_filters)
  uBlock3 = upsampling_block(uBlock2, cBlock2[1], 2*n_filters)
  uBlock4 = upsampling_block(uBlock3, cBlock1[1], n_filters)

  output = Conv2D(filters = n_classes, kernel_size = (1, 1), activation='sigmoid', padding='same')(uBlock4)

  model = Model(inputs=inputs, outputs=output)
  return model

def model_learn(model, batch_size, epoch_num, x, y):
  model.compile(loss=ev.dice_coef_loss, optimizer=RMSprop(lr=1e-4), metrics=[ev.dice_coef])
  
  history = model.fit(x, y, batch_size, epoch_num)
  model.save('./arch/unet.hdf5')


