import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.u_net_tf_v2 import UNet
# from dataset_generator import train_generator
from psuedo_label_dataset_generator import train_generator


input_size = 1024

epochs = 50
# batch_size = 2

orig_width = 1918
orig_height = 1280

threshold = 0.5

# dataset_dir = 'D:\Datasets_HDD\Carvana'
# df_train = pd.read_csv('D:\Datasets_HDD\Carvana\\train_masks.csv')
# ids_train = df_train['img'].map(lambda s: s.split('.')[0])

# ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.0, random_state=42)

# print('Training on {} samples'.format(len(ids_train_split)))
# print('Validating on {} samples'.format(len(ids_valid_split)))
#
#
# def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
#                              sat_shift_limit=(-255, 255),
#                              val_shift_limit=(-255, 255), u=0.5):
#     if np.random.random() < u:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(image)
#         hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
#         h = cv2.add(h, hue_shift)
#         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
#         s = cv2.add(s, sat_shift)
#         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
#         v = cv2.add(v, val_shift)
#         image = cv2.merge((h, s, v))
#         image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#
#     return image
#
#
# def randomShiftScaleRotate(image, mask, mask_w,
#                            shift_limit=(-0.0625, 0.0625),
#                            scale_limit=(-0.1, 0.1),
#                            rotate_limit=(-45, 45), aspect_limit=(0, 0),
#                            borderMode=cv2.BORDER_CONSTANT, u=0.5):
#     if np.random.random() < u:
#         height, width, channel = image.shape
#
#         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
#         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
#         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
#         sx = scale * aspect / (aspect ** 0.5)
#         sy = scale / (aspect ** 0.5)
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
#
#         cc = np.math.cos(angle / 180 * np.math.pi) * sx
#         ss = np.math.sin(angle / 180 * np.math.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
#
#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
#
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)
#         image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                     borderValue=(
#                                         0, 0,
#                                         0,))
#         mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         mask_w = cv2.warpPerspective(mask_w, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                      borderValue=(
#                                          0, 0,
#                                          0,))
#
#     return image, mask, mask_w
#
#
# def randomHorizontalFlip(image, mask, mask_w, u=0.5):
#     if np.random.random() < u:
#         image = cv2.flip(image, 1)
#         mask = cv2.flip(mask, 1)
#         mask_w = cv2.flip(mask_w, 1)
#
#     return image, mask, mask_w
#
#
# def train_generator():
#     while True:
#         for start in range(0, len(ids_train_split), batch_size):
#             x_batch = []
#             y_batch = []
#             w_batch = []
#             end = min(start + batch_size, len(ids_train_split))
#             ids_train_batch = ids_train_split[start:end]
#             for id in ids_train_batch.values:
#                 img = cv2.imread('D:\Datasets_HDD\Carvana\\train_hq\\{}.jpg'.format(id))
#                 img = cv2.resize(img, (input_size, input_size))
#                 mask = cv2.imread('D:\Datasets_HDD\Carvana\\output_masks\\{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (input_size, input_size))
#                 mask_w = cv2.imread('D:\Datasets_HDD\Carvana\\contour_masks\\{}_contour.png'.format(id),
#                                     cv2.IMREAD_GRAYSCALE)
#                 mask_w = cv2.resize(mask_w, (input_size, input_size))
#
#                 # cv2.imshow('i', img)
#                 # cv2.imshow('m', mask)
#                 # cv2.imshow('w', mask_w)
#                 # cv2.waitKey(0)
#                 img = randomHueSaturationValue(img,
#                                                hue_shift_limit=(0, 0),
#                                                sat_shift_limit=(-5, 5),
#                                                val_shift_limit=(-15, 15))
#                 img, mask, mask_w = randomShiftScaleRotate(img, mask, mask_w,
#                                                            # shift_limit=(-0.0625, 0.0625),
#                                                            shift_limit=(-0.0, 0.0),
#                                                            scale_limit=(-0.0, 0.0),
#                                                            rotate_limit=(-0, 0))
#                 img, mask, mask_w = randomHorizontalFlip(img, mask, mask_w)
#                 mask = np.expand_dims(mask, axis=2)
#                 mask_w = np.expand_dims(mask_w, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)
#                 w_batch.append(mask_w)
#             x_batch = np.array(x_batch, np.float32) / 255.0
#             y_batch = np.array(y_batch, np.float32) / 255.0
#             w_batch = np.array(w_batch, np.float32) / 255.0
#
#             x_batch -= 0.5
#             yield x_batch, y_batch, w_batch


# def valid_generator():
#     while True:
#         for start in range(0, len(ids_valid_split), batch_size):
#             x_batch = []
#             y_batch = []
#             end = min(start + batch_size, len(ids_valid_split))
#             ids_valid_batch = ids_valid_split[start:end]
#             for id in ids_valid_batch.values:
#                 img = cv2.imread('D:\Datasets_HDD\Carvana\\train\\{}.jpg'.format(id))
#                 img = cv2.resize(img, (input_size, input_size))
#                 mask = cv2.imread('D:\Datasets_HDD\Carvana\\output_masks\\{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (input_size, input_size))
#                 mask = np.expand_dims(mask, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)
#             x_batch = np.array(x_batch, np.float32) / 255
#             y_batch = np.array(y_batch, np.float32) / 255
#             yield x_batch, y_batch


model = UNet(input_size, is_training=True)
train_op = model.train(0.01)
input_generator = train_generator()

output = model.output_mask
output_mask_prob = tf.sigmoid(output)

tf_writer = tf.summary.FileWriter(logdir='./')
saver = tf.train.Saver()

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'model\\w_contour-365000')
    print('model loaded')

    while True:

        x_train, y_train, w_train = input_generator.__next__()

        _, loss_np, gs_np, dice_acc_np, summary_np, w_map, lr_np, \
        output_mask_np = sess.run([train_op, model.loss, model.gs, model.dice_acc,
                                                               model.merged_summary, model.loss_out, model.lr, output_mask_prob],
                                                              feed_dict={model.input: x_train,
                                                                         model.gt_mask: y_train,
                                                                         model.loss_w: w_train})
        # print(np.amax(w_map), np.amin(w_map))
        tf_writer.add_summary(summary_np, global_step=gs_np)
        print('step {:>6}, loss {:>1.5f}, dice {:>1.5f}, lr {:>1.5f}'.format(gs_np, loss_np, dice_acc_np, lr_np))

        if (gs_np + 1) % 100 == 0:
            # overlay mask img
            output_mask_img = np.zeros((model.input_size, model.input_size, 3))
            output_mask_img[:, :, 1] = np.squeeze(output_mask_np[0], axis=2) * 255.0
            car_img = (x_train[0] + 0.5) * 255.0
            show_img = 0.5 * car_img + 0.5 * output_mask_img

            # difference map
            diff_map = np.abs(y_train[0]-output_mask_np[0])
            diff_map *= 255.0
            diff_map = np.repeat(diff_map, 3, axis=2)

            verbose_img = np.concatenate((show_img, diff_map), axis=1)
            cv2.imshow('i', verbose_img.astype(np.uint8))
            cv2.waitKey(1000)
        if (gs_np + 1) % 5000 == 0:
            saver.save(sess, save_path='model/w_contour_v2', global_step=(gs_np+1), )
