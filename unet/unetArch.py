import numpy as np

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers.sexperimental import preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, UpSampling2D, concatenate
from tensorflow.keras.utils import plot_model

# unetクラスを作成
class unetArch(object):
  def __init__(self):
    self.INPUT_IMAGE_SIZE = 256
    self.INPUT_CHANNELS = 3

    # 256 x 256 x 3
    inputImage = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, self.INPUT_CHANNELS))
    # エンコーダーの作成
    # (64 x 64 x 2N)
    enc1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputImage)
    enc2 = Conv2D(64, (3, 3), padding='same', activation='relu')(enc1)

    # (128 x 128 x 4N)
    enc3 = MaxPooling2D(pool_size=(2, 2))(enc2)
    enc4 = Conv2D(128, (3, 3), padding='same',activation='relu')(enc3)
    enc5 = Conv2D(128, (3, 3), padding='same', activation='relu')(enc4)
    enc6 = Conv2D(128, (3, 3), padding='same', activation='relu')(enc5)
    #(256 x 256 x 
    enc7 = MaxPooling2D(pool_size=(2, 2))(enc6)
    enc8 = Conv2D(256, (3, 3), padding='same', activation='relu')(enc7)
    enc9 = Conv2D(256, (3, 3), padding='same', activation='relu')(enc8)
    enc10 = Conv2D(256, (3, 3), padding='same', activation='relu')(enc9)

    enc11 = MaxPooling2D(pool_size=(2, 2))(enc10)
    enc12 = Conv2D(512, (3, 3), padding='same', activation='relu')(enc11)
    enc13 = Conv2D(512, (3, 3), padding='same', activation='relu')(enc12)
    enc14 = Conv2D(512, (3, 3), padding='same', activation='relu')(enc13)

    enc15 = MaxPooling2D(pool_size=(2, 2))(enc14)
    #enc15=Dropout(0.5)(enc15)
    enc16 = Conv2D(1024, (3, 3), padding='same', activation='relu')(enc15)
    enc17 = Conv2D(1024, (3, 3), padding='same', activation='relu')(enc16)
    enc18 = Conv2D(1024, (3, 3), padding='same', activation='relu')(enc17)

    dec1 = UpSampling2D(size=(2, 2))(enc18)
    dec2 = concatenate([dec1, enc14], axis=-1)
    dec2 = Dropout(0.5)(dec2)
    dec3 = Conv2D(512, (3, 3), padding='same', activation='relu')(dec2)
    dec4 = Conv2D(512, (3, 3), padding='same', activation='relu')(dec3)

    dec5 = UpSampling2D(size=(2, 2))(dec4)
    dec6 = concatenate([dec5, enc10], axis=-1)
    dec6 = Dropout(0.5)(dec6)
    dec7 = Conv2D(256, (3, 3), padding='same', activation='relu')(dec6)
    dec8 = Conv2D(256, (3, 3), padding='same', activation='relu')(dec7)

    dec9 = UpSampling2D(size=(2, 2))(dec8)
    dec10 = concatenate([dec9, enc6], axis=-1)
    dec10 = Dropout(0.5)(dec10)
    dec11 = Conv2D(128, (3, 3), padding='same', activation='relu')(dec10)
    dec12 = Conv2D(128, (3, 3), padding='same', activation='relu')(dec11)

    dec13 = UpSampling2D(size=(2, 2))(dec12)
    dec14 = concatenate([dec13, enc2], axis=-1)
    dec14 = Dropout(0.5)(dec14)
    dec15 = Conv2D(64, (3, 3), padding='same', activation='relu')(dec14)
    dec16 = Conv2D(64, (3, 3), padding='same', activation='relu')(dec15)

    dec17 = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(dec16)

    self.unet_model = Model(input=inputImage, output=dec17)

  def build_model(self):
    return self.unet_model
