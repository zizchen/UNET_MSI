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
from losses import *

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()

        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        class_weights = load_class_weights(self)
        inputs = Input((self.img_rows, self.img_cols, 6))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(19, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(19, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['categorical_accuracy'])

        return model

    def train(self):
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        try:
            my_model = load_model('my_model_MSI.h5')
            print("loaded chk")
        except:
            my_model = self.get_unet()
            print("couldnt load chk")
        model_checkpoint = ModelCheckpoint('my_model_chk_MSI.hdf5', monitor='categorical_accuracy', verbose=1, save_best_only=True)
        tbcallback = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=1, 
            write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, 
            embeddings_layer_names=False, embeddings_metadata=False, embeddings_data=False, 
            update_freq='epoch')
        for i in range(100):
            print('step: ' + str(i))
            my_model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=5, verbose=1,
                  validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])
            my_model.save('my_model_MSI.h5', overwrite=True)
            imgs_mask_test = my_model.predict(imgs_test, batch_size=1, verbose=1)
            np.save('./results/mask_test_MSI.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/mask_test_MSI.npy')
        piclist = []
        for line in open("./results/testlist.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                    num = np.argmax(imgs[i][k][j])
                    if num == 0:
                        img[k][j] = [128, 128, 128]
                    elif num == 1:
                        img[k][j] = [128, 0, 0]
                    elif num == 2:
                        img[k][j] = [192, 192, 128]
                    elif num == 3:
                        img[k][j] = [255, 69, 0]
                    elif num == 4:
                        img[k][j] = [128, 64, 128]
                    elif num == 5:
                        img[k][j] = [60, 40, 222]
                    elif num == 6:
                        img[k][j] = [128, 128, 0]
                    elif num == 7:
                        img[k][j] = [192, 128, 128]
                    elif num == 8:
                        img[k][j] = [64, 64, 128]
                    elif num == 9:
                        img[k][j] = [64, 0, 128]
                    elif num == 10:
                        img[k][j] = [64, 64, 0]
                    elif num == 11:
                        img[k][j] = [33, 128, 64]
                    elif num == 12:
                        img[k][j] = [90, 10, 22]
                    elif num == 13:
                        img[k][j] = [32, 155, 192]
                    elif num == 14:
                        img[k][j] = [0, 192, 0]
                    elif num == 15:
                        img[k][j] = [192, 0, 192]
                    elif num == 16:
                        img[k][j] = [0, 255, 192]
                    elif num == 17:
                        img[k][j] = [255, 128, 0]
                    elif num == 18:
                        img[k][j] = [75, 10, 50]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img)

if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
