import tensorflow as tf
from model.u_net_tf_v3 import UNet as UNet1
from model.u_net_tf import UNet as UNet2
from utils import scale_adjust, hsv_adjust, shift_adjust, dice
import argparse
import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('model_file', type=str, default='model/w_contour_v3-380000', nargs='?')
# parser.add_argument('model_file', type=str, default='model/backup/w_contour_v3-280000', nargs='?')
# parser.add_argument('model_file', type=str, default='model/w_contour-365000', nargs='?')
parser.add_argument('model_file2', type=str, default='model/backup/w_contour_v2-545000', nargs='?')
parser.add_argument('input_size', type=int, default=1280, nargs='?')
parser.add_argument('with_flip', type=bool, default=True, nargs='?')
parser.add_argument('with_scale', type=bool, default=False, nargs='?')
parser.add_argument('with_hsv', type=bool, default=True, nargs='?')
parser.add_argument('with_shift', type=bool, default=False, nargs='?')
parser.add_argument('with_rotate', type=bool, default=False, nargs='?')
parser.add_argument('--train_img_dir', type=str, default='/home/tim_ho/Documents/Dataset_SSD/Carvana/train_hq')
parser.add_argument('--gt_mask_dir', type=str, default='/home/tim_ho/Documents/Dataset_SSD/Carvana/png_masks/')
args = parser.parse_args()


def test_with_augmentation():
    model = UNet1(args.input_size, is_training=False)
    output_mask = model.output_mask
    output_prob = tf.sigmoid(output_mask)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, args.model_file)

    train_img_dir = args.train_img_dir
    gt_mask_dir = args.gt_mask_dir
    gt_mask_list = os.listdir(gt_mask_dir)
    start_idx = np.random.randint(5000)
    gt_mask_list = gt_mask_list[start_idx:start_idx+20]

    augmentation_combinations = []
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(2):
                        augmentation_combinations.append([i0, i1, i2, i3, i4])

    best_score = 0

    # search thresholds
    while True:
        thresh = 0
        while thresh < 0.2 or thresh > 0.7:
            thresh = np.random.rand()

        # search possible augmentation combinations
        for combination in [[0, 0, 1, 0, 0]]:#augmentation_combinations:
            dice_sum = 0
            ens_dice_sum = 0
            for img_name in gt_mask_list:

                args.with_shift, args.with_scale, args.with_flip, args.with_rotate, args.with_hsv = \
                combination[0], combination[1], combination[2], combination[3], combination[4]
                # args.with_shift = True if np.random.rand() > 0.5 else False
                # args.with_scale = True if np.random.rand() > 0.5 else False
                # args.with_flip = True if np.random.rand() > 0.5 else False
                # args.with_rotate = True if np.random.rand() > 0.5 else False
                # args.with_hsv = True if np.random.rand() > 0.5 else False
                params = {'with_shift': args.with_shift,
                          'with_scale': args.with_scale,
                          'with_flip': args.with_flip,
                          'with_hsv': args.with_hsv,
                          'with_rotate': args.with_rotate}


                train_img_name = img_name.split('_mask')[0] + '.jpg'
                img_path = os.path.join(train_img_dir, train_img_name)
                gt_mask_path = os.path.join(gt_mask_dir, img_name)


                img = cv2.imread(img_path)
                img = cv2.resize(img, (args.input_size, args.input_size))
                gt_mask = cv2.imread(gt_mask_path)
                gt_mask = gt_mask[:, :, 0] / 255.0

                # create test augmented batch
                test_batch = []
                test_batch.append(img)
                idx = 1
                idx_hsv_start = 0
                idx_scale_start = 0
                idx_shift_start = 0
                idx_rotate_start = 0
                idx_flip_start = 0

                if args.with_flip:
                    idx_flip_start = idx
                    idx += 1
                    test_batch.append(cv2.flip(img, 1))

                if args.with_hsv:
                    idx_hsv_start = idx
                    idx += 2
                    test_batch.append(hsv_adjust(img, 5, 10, 10))
                    test_batch.append(hsv_adjust(img, -5, -10, -10))

                if args.with_scale:
                    idx_scale_start = idx
                    idx += 2
                    test_batch.append(scale_adjust(img, 1.1))
                    test_batch.append(scale_adjust(img, 0.9))

                if args.with_shift:
                    idx_shift_start = idx
                    idx += 4
                    test_batch.append(shift_adjust(img, 0.03, 0))
                    test_batch.append(shift_adjust(img, -0.03, 0))
                    test_batch.append(shift_adjust(img, 0, -0.03))
                    test_batch.append(shift_adjust(img, 0, 0.03))

                test_batch_input = np.array(test_batch, np.float32) / 255.0
                test_batch_input -= 0.5

                # forward
                output_prob_np = sess.run([output_prob], feed_dict={model.input: test_batch_input})
                output_prob_np = output_prob_np[0]

                # flip back
                if args.with_flip:
                    tmp = output_prob_np[idx_flip_start, :, :, :]
                    tmp = np.fliplr(tmp)
                    output_prob_np[idx_flip_start, :, :, 0] = tmp[:, :, 0]

                # scale back
                if args.with_scale:
                    tmp = output_prob_np[idx_scale_start, : ,:, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1/1.1))
                    output_prob_np[idx_scale_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_scale_start+1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1 / 0.9))
                    output_prob_np[idx_scale_start+1, :, :, 0] = tmp[:, :, 0]

                # shift back
                if args.with_shift:
                    tmp = output_prob_np[idx_shift_start, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, -0.03, 0)
                    output_prob_np[idx_shift_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0.03, 0)
                    output_prob_np[idx_shift_start+1, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+2, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, 0.03)
                    output_prob_np[idx_shift_start+2, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+3, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, -0.03)
                    output_prob_np[idx_shift_start+3, :, :, 0] = tmp[:, :, 0]

                # single output
                single_pred = output_prob_np[0]
                single_pred_prob = cv2.resize(single_pred, (1918, 1280))
                single_pred_mask = (single_pred_prob > thresh).astype(np.uint8)

                # ensemble output
                ensemble_output = np.mean(output_prob_np, axis=0)
                ensemble_output = cv2.resize(ensemble_output, (1918, 1280))

                # thresh = 0
                # while thresh < 0.05 or thresh > 0.6:
                #     thresh = np.random.rand()
                ensemble_output_mask = (ensemble_output > thresh).astype(np.uint8)

                dice_sum += dice(single_pred_mask, gt_mask)
                ens_dice_sum += dice(ensemble_output_mask, gt_mask)
                # print('Improve = {} | Oirg = {:>2.6f} | Ens = {:>2.6f}'.format((dice(single_pred, gt_mask) < dice(ensemble_output_mask, gt_mask)),
                #                                                    dice(single_pred, gt_mask),
                #                                                    dice(ensemble_output_mask, gt_mask)))

            if (ens_dice_sum / 20.0) > best_score and (ens_dice_sum > dice_sum):
                best_score = (ens_dice_sum / 20.0)
                print('thresh {}'.format(thresh))
                print('Best Score {:>2.6f} | Orig Score {:>2.6f} | {}'.format(best_score, dice_sum / 20.0, params))
                print('')
                # print(params)
                # print('')


def test_with_multimodel_augmentation():

    graph1 = tf.Graph()
    with graph1.as_default():
        model1 = UNet1(args.input_size, is_training=False)
        output_mask1 = model1.output_mask
        output_prob1 = tf.sigmoid(output_mask1)

        saver = tf.train.Saver()

        sess1 = tf.Session(graph=graph1)
        sess1.run(tf.global_variables_initializer())
        saver.restore(sess1, args.model_file)

    graph2 = tf.Graph()
    with graph2.as_default():
        model2 = UNet2(args.input_size, is_training=False)
        output_mask2 = model2.output_mask
        output_prob2 = tf.sigmoid(output_mask2)

        saver = tf.train.Saver()

        sess2 = tf.Session(graph=graph2)
        sess2.run(tf.global_variables_initializer())
        saver.restore(sess2, args.model_file2)

    util_graph = tf.Graph()
    with util_graph.as_default():
        resize_holder = tf.placeholder(dtype=tf.float32, shape=[None, args.input_size, args.input_size, 1])
        ensemble_output_op = tf.reduce_mean(tf.image.resize_bilinear(resize_holder, (1280, 1918)), axis=0)
        util_sess = tf.Session(graph=util_graph)
        util_sess.run(tf.global_variables_initializer())


    train_img_dir = args.train_img_dir
    gt_mask_dir = args.gt_mask_dir
    gt_mask_list = os.listdir(gt_mask_dir)
    start_idx = np.random.randint(5000)
    start_idx = 100
    gt_mask_list = gt_mask_list[start_idx:start_idx+150:10]

    augmentation_combinations = []
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(2):
                        augmentation_combinations.append([i0, i1, i2, i3, i4])

    best_score = 0
    best_orig_score = 0

    # search thresholds
    while True:
        thresh = 0
        while thresh < 0.3 or thresh > 0.6:
            thresh = np.random.rand()

        # search possible augmentation combinations
        for combination in augmentation_combinations:
            dice_sum = 0
            ens_dice_sum = 0
            for img_name in gt_mask_list:

                args.with_shift, args.with_scale, args.with_flip, args.with_rotate, args.with_hsv = \
                combination[0], combination[1], combination[2], combination[3], combination[4]
                # args.with_shift = True if np.random.rand() > 0.5 else False
                # args.with_scale = True if np.random.rand() > 0.5 else False
                # args.with_flip = True if np.random.rand() > 0.5 else False
                # args.with_rotate = True if np.random.rand() > 0.5 else False
                # args.with_hsv = True if np.random.rand() > 0.5 else False
                params = {'with_shift': args.with_shift,
                          'with_scale': args.with_scale,
                          'with_flip': args.with_flip,
                          'with_hsv': args.with_hsv,
                          'with_rotate': args.with_rotate}


                train_img_name = img_name.split('_mask')[0] + '.jpg'
                img_path = os.path.join(train_img_dir, train_img_name)
                gt_mask_path = os.path.join(gt_mask_dir, img_name)


                img = cv2.imread(img_path)
                img = cv2.resize(img, (args.input_size, args.input_size))
                gt_mask = cv2.imread(gt_mask_path)
                gt_mask = gt_mask[:, :, 0] / 255.0

                # create test augmented batch
                test_batch = []
                test_batch.append(img)
                idx = 1
                idx_hsv_start = 0
                idx_scale_start = 0
                idx_shift_start = 0
                idx_rotate_start = 0
                idx_flip_start = 0

                if args.with_flip:
                    idx_flip_start = idx
                    idx += 1
                    test_batch.append(cv2.flip(img, 1))

                if args.with_hsv:
                    idx_hsv_start = idx
                    idx += 2
                    test_batch.append(hsv_adjust(img, 5, 10, 10))
                    test_batch.append(hsv_adjust(img, -5, -10, -10))

                if args.with_scale:
                    idx_scale_start = idx
                    idx += 2
                    test_batch.append(scale_adjust(img, 1.1))
                    test_batch.append(scale_adjust(img, 0.9))

                if args.with_shift:
                    idx_shift_start = idx
                    idx += 4
                    test_batch.append(shift_adjust(img, 0.03, 0))
                    test_batch.append(shift_adjust(img, -0.03, 0))
                    test_batch.append(shift_adjust(img, 0, -0.03))
                    test_batch.append(shift_adjust(img, 0, 0.03))

                test_batch_input = np.array(test_batch, np.float32) / 255.0
                test_batch_input -= 0.5

                # forward model 1
                output_prob_np = sess1.run([output_prob1], feed_dict={model1.input: test_batch_input})
                output_prob_np = output_prob_np[0]

                # forward model 2
                output_prob_np2 = sess2.run([output_prob2], feed_dict={model2.input: test_batch_input})
                output_prob_np2 = output_prob_np2[0]

                ## model 1 adjust
                # flip back
                if args.with_flip:
                    tmp = output_prob_np[idx_flip_start, :, :, :]
                    tmp = np.fliplr(tmp)
                    output_prob_np[idx_flip_start, :, :, 0] = tmp[:, :, 0]

                # scale back
                if args.with_scale:
                    tmp = output_prob_np[idx_scale_start, : ,:, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1/1.1))
                    output_prob_np[idx_scale_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_scale_start+1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1 / 0.9))
                    output_prob_np[idx_scale_start+1, :, :, 0] = tmp[:, :, 0]

                # shift back
                if args.with_shift:
                    tmp = output_prob_np[idx_shift_start, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, -0.03, 0)
                    output_prob_np[idx_shift_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0.03, 0)
                    output_prob_np[idx_shift_start+1, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+2, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, 0.03)
                    output_prob_np[idx_shift_start+2, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np[idx_shift_start+3, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, -0.03)
                    output_prob_np[idx_shift_start+3, :, :, 0] = tmp[:, :, 0]

                ## model 2 adjust
                # flip back
                if args.with_flip:
                    tmp = output_prob_np2[idx_flip_start, :, :, :]
                    tmp = np.fliplr(tmp)
                    output_prob_np2[idx_flip_start, :, :, 0] = tmp[:, :, 0]

                # scale back
                if args.with_scale:
                    tmp = output_prob_np2[idx_scale_start, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1 / 1.1))
                    output_prob_np2[idx_scale_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np2[idx_scale_start + 1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = scale_adjust(tmp, (1 / 0.9))
                    output_prob_np2[idx_scale_start + 1, :, :, 0] = tmp[:, :, 0]

                # shift back
                if args.with_shift:
                    tmp = output_prob_np2[idx_shift_start, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, -0.03, 0)
                    output_prob_np2[idx_shift_start, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np2[idx_shift_start + 1, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0.03, 0)
                    output_prob_np2[idx_shift_start + 1, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np2[idx_shift_start + 2, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, 0.03)
                    output_prob_np2[idx_shift_start + 2, :, :, 0] = tmp[:, :, 0]

                    tmp = output_prob_np2[idx_shift_start + 3, :, :, :]
                    tmp = np.repeat(tmp, 3, axis=2)
                    tmp = shift_adjust(tmp, 0, -0.03)
                    output_prob_np2[idx_shift_start + 3, :, :, 0] = tmp[:, :, 0]


                # single output
                single_pred = output_prob_np[0]
                single_pred = cv2.resize(single_pred, (1918, 1280))
                single_pred_mask = (single_pred > thresh).astype(np.uint8)
                single_pred2 = output_prob_np2[0]
                single_pred2 = cv2.resize(single_pred2, (1918, 1280))
                single_pred2_mask = (single_pred2 > thresh).astype(np.uint8)

                # ensemble output
                # ensemble_output = np.mean(output_prob_np, axis=0)
                # ensemble_output = cv2.resize(ensemble_output, (1918, 1280))
                ensemble_output = util_sess.run([ensemble_output_op], feed_dict={resize_holder: output_prob_np})
                ensemble_output = ensemble_output[0]
                ensemble_output = np.squeeze(ensemble_output, axis=2)

                # ensemble output2
                # ensemble_output2 = np.mean(output_prob_np2, axis=0)
                # ensemble_output2 = cv2.resize(ensemble_output2, (1918, 1280))
                ensemble_output2 = util_sess.run([ensemble_output_op], feed_dict={resize_holder: output_prob_np2})
                ensemble_output2 = ensemble_output2[0]
                ensemble_output2 = np.squeeze(ensemble_output2, axis=2)

                # fuse output
                # ensemble_output = (ensemble_output + ensemble_output2) / 2.0
                ensemble_output = (single_pred + single_pred2 + ensemble_output + ensemble_output2) / 4.0

                # thresh = 0
                # while thresh < 0.05 or thresh > 0.6:
                #     thresh = np.random.rand()
                ensemble_output_mask = (ensemble_output > thresh).astype(np.uint8)

                dice_sum += dice(single_pred_mask, gt_mask)
                ens_dice_sum += dice(ensemble_output_mask, gt_mask)
                # print('Improve = {} | Oirg = {:>2.6f} | Ens = {:>2.6f}'.format((dice(single_pred_mask, gt_mask) < dice(ensemble_output_mask, gt_mask)),
                #                                                    dice(single_pred_mask, gt_mask),
                #                                                    dice(ensemble_output_mask, gt_mask)))
                # print('model1 {} | model2 {}'.format(dice(single_pred_mask, gt_mask), dice(single_pred2_mask, gt_mask)))
                # print('')

            if (dice_sum / len(gt_mask_list)) > best_orig_score:
                best_orig_score = (dice_sum / len(gt_mask_list))
            # print((dice_sum / len(gt_mask_list)))

            if (ens_dice_sum / len(gt_mask_list)) > best_score and (ens_dice_sum / len(gt_mask_list)) > best_orig_score:
                best_score = (ens_dice_sum / len(gt_mask_list))
                print('thresh {}'.format(thresh))
                print('Best Score {:>2.6f} | Orig Score {:>2.6f} | {}'.format(best_score, dice_sum / len(gt_mask_list), params))
                print('')
                # print(params)
                # print('')









if __name__ == '__main__':
    test_with_multimodel_augmentation()
    # test_with_augmentation()