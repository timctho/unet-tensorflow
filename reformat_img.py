from scipy import misc
import numpy as np
import os

dataset_dir = 'D:\Datasets_HDD\Carvana\\train_masks\\train_masks'
output_dir = 'D:\Datasets_HDD\Carvana\\output_masks'

for i in os.listdir(dataset_dir):
    img_file = dataset_dir + '\\' + i
    img = misc.imread(img_file)
    img_mean = np.mean(img, axis=2)
    img_out = np.expand_dims(img_mean, axis=2)
    output_file = output_dir + '\\' + i.split('.gif')[0] + '.png'
    misc.imsave(output_file, img_mean)
