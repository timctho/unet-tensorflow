import numpy as np
import cv2
import os
from scipy import misc

dataset_dir = 'D:\Datasets_HDD\Carvana\\output_masks\\'
output_dir = 'D:\Datasets_HDD\Carvana\\contour_masks'

for i in os.listdir(dataset_dir):
    img = cv2.imread(dataset_dir + i)
    img_gray = img[:, :, 0] / 255
    img_gray = img_gray.astype(np.uint8)
    im2, c, hie = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros(shape=(1280, 1918, 3))
    cv2.drawContours(img, c, -1, (0, 255, 0), 3)
    img = img[:, :, 1]
    img_save_path = output_dir + '\\' + i.split('mask')[0] + 'contour.png'
    print(img_save_path)
    misc.imsave(img_save_path, img)
