import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import math
from scipy.io import loadmat
import os # path join
from sklearn.feature_extraction import image
from scipy import misc
import tifffile

file_path = './data/rit18_data.mat'

dataset = loadmat(file_path)
no_train_patches = 300
no_test_patches = 30
seed = 99

#Load Training Data and Labels
train_data = dataset['train_data']
train_mask = train_data[-1]
train_data = train_data[0:6]
train_labels = dataset['train_labels']

#Load Validation Data and Labels
val_data = dataset['val_data']
val_mask = val_data[-1]
val_data = val_data[0:6]
val_labels = dataset['val_labels']

#Load Test Data
test_data = dataset['test_data']
test_mask = test_data[-1]
test_data = test_data[:6]

train_data_array = np.zeros([9393, 5642, 6])
train_data_array[0:9392, 0:5641, 0] = train_data[0, 0:9392, 0:5641]
train_data_array[0:9392, 0:5641, 1] = train_data[1, 0:9392, 0:5641]
train_data_array[0:9392, 0:5641, 2] = train_data[2, 0:9392, 0:5641]
train_data_array[0:9392, 0:5641, 3] = train_data[3, 0:9392, 0:5641]
train_data_array[0:9392, 0:5641, 4] = train_data[4, 0:9392, 0:5641]
train_data_array[0:9392, 0:5641, 5] = train_data[5, 0:9392, 0:5641]

test_data_array = np.zeros([8833, 5642, 6])
test_data_array[0:8833, 0:5641, 0] = val_data[0, 0:8833, 0:5641]
test_data_array[0:8833, 0:5641, 1] = val_data[1, 0:8833, 0:5641]
test_data_array[0:8833, 0:5641, 2] = val_data[2, 0:8833, 0:5641]
test_data_array[0:8833, 0:5641, 3] = val_data[3, 0:8833, 0:5641]
test_data_array[0:8833, 0:5641, 4] = val_data[4, 0:8833, 0:5641]
test_data_array[0:8833, 0:5641, 5] = val_data[5, 0:8833, 0:5641]

train_patches = image.extract_patches_2d(train_data_array, (512,512), max_patches=no_train_patches, 
	random_state = seed)
label_patches = image.extract_patches_2d(train_labels, (512,512), max_patches=no_train_patches, 
	random_state = seed)
test_patches = image.extract_patches_2d(test_data_array, (512,512), max_patches=no_test_patches, 
	random_state = seed)
test_label_patches = image.extract_patches_2d(val_labels, (512,512), max_patches=no_test_patches, 
	random_state = seed)

for i in range(no_train_patches):
	tifffile.imsave('./data/train/image/%s.tiff' % (i), train_patches[i,:,:,:])
	tifffile.imsave('./data/train/label/%s.tiff' % (i), label_patches[i,:,:])

for i in rangno_test_patches):
	tifffile.imsave('./data/test/image/%s.tiff' % (i), test_patches[i,:,:,:])
	tifffile.imsave('./data/test/label/%s.tiff' % (i), test_label_patches[i,:,:])

