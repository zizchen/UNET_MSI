from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from skimage import io

class dataProcess(object):
    def __init__(self, out_rows, out_cols, train_path="data/train/image", train_label="data/train/label",
                 val_path="data/val", val_label="data/valannot",
                 test_path="data/test/image", test_label='data/test/label', npy_path="./npydata", img_type="tiff"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.train_path = train_path
        self.train_label = train_label
        self.img_type = img_type
        self.val_path = val_path
        self.val_label = val_label
        self.test_path = test_path
        self.test_label = test_label
        self.npy_path = npy_path

    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 19])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                #print(label[i][j])
                x[i, j, int(label[i][j])] = 1  # 属于第m类，第三维m处值为1
                #print(x[i][j])
        #print(x)
        return x

    def class_weights(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 1])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                #print(label[i][j])
                x[i, j, 0] = label[i][j][0]  # 属于第m类，第三维m处值为1
                #print(x[i][j])
        #print(x)
        return x

    def create_train_data(self):
        i = 0
        print('Creating training images...')
        imgs = sorted(glob.glob(self.train_path+"/*."+self.img_type))
        labels = sorted(glob.glob(self.train_label+"/*."+self.img_type))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 6), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, 19), dtype=np.uint8)
        imglabelsweights = np.ndarray((len(labels), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        print(len(imgs), len(labels))

        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]
            img = io.imread(imgpath)
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            weights = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = self.label2class(img_to_array(label))
            weights = self.class_weights(img_to_array(weights))
            imgdatas[i] = img
            imglabels[i] = label
            imglabelsweights[i] = weights
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save(self.npy_path + '/train_MSI.npy', imgdatas)
        np.save(self.npy_path + '/mask_train_MSI.npy', imglabels)
        np.save(self.npy_path + '/mask_weights_MSI.npy', imglabelsweights)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = sorted(glob.glob(self.test_path + "/*." + self.img_type))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 6), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = io.imread(testpath)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = './npydata/testlist.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save(self.npy_path + '/test_MSI.npy', imgdatas)
        print('Saving to test.npy files done.')

    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/train_MSI.npy")
        imgs_mask_train = np.load(self.npy_path + "/mask_train_MSI.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        #imgs_train /= 255
        #imgs_mask_train /= 18
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('load test images...')
        imgs_test = np.load(self.npy_path + "/test_MSI.npy")
        imgs_test = imgs_test.astype('float32')
        #imgs_test /= 255
        return imgs_test

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/mask_test_MSI.npy')
        piclist = []
        for line in open("./npydata/testlist.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):
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


if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
