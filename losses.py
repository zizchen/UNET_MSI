# -*- coding:utf-8 -*-
import keras
from keras.models import *
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import array_to_img
import cv2
from data import *
import matplotlib.pyplot as plt
from mayavi import mlab
import tensorflow as tf

def weighted_pixelwise_crossentropy(class_weights):
    
    def lossFunction(y_true, y_pred):
        #epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        #y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        print(y_true)
        print(y_pred)
        print(tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights)))
        return tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), 1000000*class_weights))

    return lossFunction

def ignore_cat_0(class_weights):
    def ignore_category_0(ytrue, ypred):
        #epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        #y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        #return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))
        #cat_cross_object = categorical_crossentropy(ytrue, ypred) #calculates unweighted categorical_crossentropy as found in keras
        #weighted_loss = 0
        #cat_cross_object = cat_cross_object*(1-ytrue[:, :, :, 0]) #removes 0 category
        #for i in range(len(class_weights)-1):
        #    weighted_loss += tf.reduce_sum(tf.multiply(categorical_crossentropy(ytrue, ypred),((1/class_weights[i])*ytrue[:,:,:,i+1])))
        #return weighted_loss

        return (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue, ypred)
        #return (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue, ypred)
    return ignore_category_0

def load_class_weights(self):
    print('loading class_weights...')
    imgs_mask_weights = np.load("./npydata" + "/mask_weights_MSI.npy")
    imgs_mask_weights = imgs_mask_weights.astype('float32')

    class_weights = [1/np.count_nonzero(imgs_mask_weights == 0),
    1/np.count_nonzero(imgs_mask_weights == 1),
    1/np.count_nonzero(imgs_mask_weights == 2),
    1/np.count_nonzero(imgs_mask_weights == 3),
    1/np.count_nonzero(imgs_mask_weights == 4),
    1/np.count_nonzero(imgs_mask_weights == 5),
    1/np.count_nonzero(imgs_mask_weights == 6),
    1/np.count_nonzero(imgs_mask_weights == 7),
    1/np.count_nonzero(imgs_mask_weights == 8),
    1/np.count_nonzero(imgs_mask_weights == 9),
    1/np.count_nonzero(imgs_mask_weights == 10),
    1/np.count_nonzero(imgs_mask_weights == 11),
    1/np.count_nonzero(imgs_mask_weights == 12),
    1/np.count_nonzero(imgs_mask_weights == 13),
    1/np.count_nonzero(imgs_mask_weights == 14),
    1/np.count_nonzero(imgs_mask_weights == 15),
    1/np.count_nonzero(imgs_mask_weights == 16),
    1/np.count_nonzero(imgs_mask_weights == 17),
    1/np.count_nonzero(imgs_mask_weights == 18)]
    print(class_weights)

    return imgs_mask_weights