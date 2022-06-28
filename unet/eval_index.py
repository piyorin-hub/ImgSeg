import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import Model

@tf.function
def dice_coef(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  intersection = K.sum(y_true * y_pred)

  if (K.sum(y_true) + K.sum(y_pred) == 0):
    return 1.0
  else:
    return (2.0 * intersection)/ (K.sum(y_true) + K.sum(y_pred))

@tf.function
def dice_coef_loss(y_true, y_pred):
  return 1.0 - dice_coef(y_true, y_pred)
