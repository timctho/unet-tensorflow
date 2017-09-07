import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

input_h = 1280
input_w = 1280

epochs = 50
batch_size = 4

orig_width = 1918
orig_height = 1280

threshold = 0.5

dataset_dir = '/home/tim_ho/Documents/Dataset_SSD/Carvana'
df_train = pd.read_csv('/home/tim_ho/Documents/Dataset_SSD/Carvana/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.0, random_state=42)


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask, mask_w,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        mask_w = cv2.warpPerspective(mask_w, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                     borderValue=(
                                         0, 0,
                                         0,))

    return image, mask, mask_w


def randomHorizontalFlip(image, mask, mask_w, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        mask_w = cv2.flip(mask_w, 1)

    return image, mask, mask_w



def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            w_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('/home/tim_ho/Documents/Dataset_SSD/Carvana/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_w, input_h))
                mask = cv2.imread('/home/tim_ho/Documents/Dataset_SSD/Carvana/png_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_w, input_h))
                # mask_w = cv2.imread('/home/tim_ho/Documents/Dataset_SSD/Carvana/contour_masks/{}_contour.png'.format(id),
                #                     cv2.IMREAD_GRAYSCALE)
                # mask_w = cv2.resize(mask_w, (input_w, input_h))


                prev_prob_mask = np.load('/media/tim_ho/HDD1/Projects/kaggle-carvana/unet-pytorch/predictions/unet1280-35/{}.npy'.format(id))
                prob_img = cv2.resize(prev_prob_mask, (input_w, input_h))
                binarize_mask = (prob_img > 0.3).astype(np.uint8)
                im2, c, hie = cv2.findContours(binarize_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_img = np.zeros(shape=(input_h, input_w, 3))
                cv2.drawContours(contour_img, c, -1, (255, 255, 255), 20)
                contour_img /= 255.0

                # rgb edge
                rgb_edge_img = np.multiply(contour_img, img)

                # prev pred edge
                prev_pred_edge_img = np.multiply(contour_img[:, :, 0], prob_img)
                prev_pred_edge_img = np.expand_dims(prev_pred_edge_img, axis=2)

                # gt mask edge
                mask = np.expand_dims(mask, axis=2)
                gt_edge_img = np.multiply(contour_img, mask)
                # cv2.imshow('', gt_edge_img)
                # cv2.imshow('contour', contour_img)
                # cv2.waitKey(0)
                tmp = np.multiply(gt_edge_img/255.0, img)
                print(gt_edge_img.shape, img.shape)
                cv2.imshow('img', img)
                cv2.imshow('', np.multiply(gt_edge_img/255.0, img).astype(np.uint8))
                cv2.waitKey(0)

                concat_input = np.concatenate((rgb_edge_img, prev_pred_edge_img * 255.0), axis=2)







                # img = randomHueSaturationValue(img,
                #                                hue_shift_limit=(-2, 2),
                #                                sat_shift_limit=(-10, 10),
                #                                val_shift_limit=(-10, 10))
                # img, mask, mask_w = randomShiftScaleRotate(img, mask, mask_w,
                #                                            shift_limit=(-0.0625, 0.0625),
                #                                            # shift_limit=(-0.0, 0.0),
                #                                            scale_limit=(-0.1, 0.1),
                #                                            rotate_limit=(-0, 0))
                # img, mask, mask_w = randomHorizontalFlip(img, mask, mask_w)

                # mask_w = np.expand_dims(mask_w, axis=2)
                x_batch.append(concat_input)
                y_batch.append(gt_edge_img)
                # w_batch.append(mask_w)
            x_batch = np.array(x_batch, np.float32) #/ 255.0
            y_batch = np.array(y_batch, np.float32) #/ 255.0

            # x_batch -= 0.5
            # w_batch = np.array(w_batch, np.float32) / 255.0
            yield x_batch, y_batch # , w_batch

if __name__ == '__main__':
    data_generator = train_generator()
    while True:
        x_batch, y_batch = data_generator.next()
        print(x_batch.shape, y_batch.shape)
        for i in range(y_batch.shape[0]):
            rgb_img = np.multiply(x_batch[i, : ,:, 0:3], y_batch[i]/255.0)
            prev_pred = x_batch[i ,: ,:, 3]

            cv2.imshow('prev', np.abs(prev_pred-y_batch[i, :, :, 0]))
            cv2.imshow('', y_batch[i])
            cv2.imshow('rgb', rgb_img)
            cv2.waitKey(0)
