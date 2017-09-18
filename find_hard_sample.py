import tensorflow as tf
import os
import argparse
import cv2
import numpy as np
from shutil import copyfile

from model.u_net_tf import UNet
from utils import dice

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='model/w_contour-360000')
parser.add_argument('--input_size', default=1024)
parser.add_argument('--input_dir', default='/home/tim_ho/Documents/Dataset_SSD/Carvana/train')
parser.add_argument('--gt_dir', default='/home/tim_ho/Documents/Dataset_SSD/Carvana/png_masks/')
parser.add_argument('--hard_sample_dir', default='/home/tim_ho/Documents/Dataset_SSD/Carvana/hard_samples')
parser.add_argument('--thresh', default=0.347)
parser.add_argument('--hard_thresh', default=0.997)
args = parser.parse_args()


def find_hard_sample():

    # create dir
    if not os.path.exists(args.hard_sample_dir):
        os.mkdir(args.hard_sample_dir)
    hard_sample_dir_hq = args.hard_sample_dir + '_hq'
    if not os.path.exists(hard_sample_dir_hq):
        os.mkdir(hard_sample_dir_hq)


    model = UNet(args.input_size, False)
    output_logits = model.output_mask
    output_prob = tf.sigmoid(output_logits)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_file)

    img_count = 0
    hard_img_count = 0
    dice_sum = 0
    gt_list = os.listdir(args.gt_dir)

    for gt_mask_name in gt_list:
        img_count += 1
        img_name = gt_mask_name.split('_mask')[0] + '.jpg'
        gt_mask_path = os.path.join(args.gt_dir, gt_mask_name)

        # read in gt mask
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask = gt_mask[:, :, 0] / 255.0

        # read in test img
        img_path = os.path.join(args.input_dir, img_name)
        orig_img = cv2.imread(img_path)
        img = cv2.resize(orig_img, (args.input_size, args.input_size))
        img = img.astype(np.float32)
        img /= 255.0
        img -= 0.5
        img = np.expand_dims(img, axis=0)

        output_prob_np = sess.run([output_prob], feed_dict={model.input: img})
        output_prob_np = output_prob_np[0]

        output_prob_np = cv2.resize(output_prob_np[0], (1918, 1280))
        output_mask = (output_prob_np > args.thresh).astype(np.uint8)
        dice_acc = dice(output_mask, gt_mask)
        dice_sum += dice_acc

        if dice_acc < args.hard_thresh:
            hard_img_count += 1
            print('dice {}'.format(dice_acc))
            print('hard img ratio {} / {}'.format(hard_img_count, img_count))
            input_dir_hq = args.input_dir + '_hq'
            hard_sample_dir_hq = args.hard_sample_dir + '_hq'

            src = os.path.join(args.input_dir, img_name)
            dst = os.path.join(args.hard_sample_dir, img_name)
            src_hq = os.path.join(input_dir_hq, img_name)
            dst_hq = os.path.join(hard_sample_dir_hq, img_name)

            copyfile(src, dst)
            copyfile(src_hq, dst_hq)
    print('mean dice {}'.format(dice_sum / len(gt_list)))


if __name__ == '__main__':
    find_hard_sample()