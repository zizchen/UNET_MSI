from unet import *
from data import *
from losses import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


myunet = myUnet()
model = myunet.get_unet()
model.load_weights('my_model_chk_MSI.hdf5')

imgs_train, imgs_mask_train, imgs_test = myunet.load_data()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('./results/mask_test_MSI.npy', imgs_mask_test)

myunet.save_img()

img=mpimg.imread('./results/image/0.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/1.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/2.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/3.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/4.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/5.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/6.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/7.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/8.png')
plt.imshow(img)
plt.show()
img=mpimg.imread('./results/image/9.png')
plt.imshow(img)
plt.show()
