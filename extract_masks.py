import numpy as np
import pandas as pd
from scipy import misc


def decode_rle(rle_str, img_h, img_w):
    temp = np.zeros(img_h * img_w)
    rle_str = rle_str.split(' ')
    for i in range(0, len(rle_str), 2):
        start = int(rle_str[i])
        steps = int(rle_str[i + 1])
        temp[start:start + steps] = 1
    return np.reshape(temp, (img_h, img_w))


def main():
    masks_path = 'submit/submission/submission.csv'
    # masks_path = 'input/train_masks.csv'
    img_save_path = 'submit_masks\\'
    masks_file = pd.read_csv(masks_path)

    for i in range(len(masks_file)):
        mask = decode_rle(masks_file['rle_mask'][i], 1280, 1918)

        img_path = img_save_path + masks_file['img'][i].split('.jpg')[0] + '_mask.png'
        misc.imsave(img_path, mask)


if __name__ == '__main__':
    main()
