import numpy as np
import tensorflow as tf
import os
import threading
import cv2
import time

best_score = {'thresh': 0, 'dice': 0, 'use_flip': False}
mutex = threading.Lock()

def dice(pred_mask, gt_mask):
    smooth = 1e-7
    flat_pred_mask = np.reshape(pred_mask, (1, -1))
    flat_gt_mask = np.reshape(gt_mask, (1, -1))
    return (2 * np.sum(np.multiply(flat_pred_mask, flat_gt_mask)) + smooth) / (np.sum(flat_pred_mask) + np.sum(flat_gt_mask) + smooth)


def search_thresh(thread_name):

    global best_score
    time.sleep(1)

    input_w = 1918
    input_h = 1280
    pred_npy_dir = 'D:\Projects\kaggle-carvana\model_training_predictions\w_contour-360000'
    flip_pred_npy_dir = 'D:\Projects\kaggle-carvana\model_training_predictions\w_contour-360000'
    gt_mask_dir = 'D:\Datasets_HDD\Carvana\output_masks'



    # start trying threshold
    while True:
        # random threshold
        thresh = 0
        while thresh < 0.2 or thresh > 0.6:
            thresh = np.random.rand()

        # use flip img or not
        use_flip = True #if np.random.rand() > 0.5 else False

        # compute dice
        dice_sum = 0
        img_count = 0
        for gt_img_path in os.listdir(gt_mask_dir):
            if (img_count + 1) % 500 == 0:
                print('{} img {}'.format(thread_name, img_count+1))
            img_count += 1

            img_name = gt_img_path.split('_mask')[0]

            # data paths
            pred_npy_path = os.path.join(pred_npy_dir, img_name+'.jpg.npy')
            flip_pred_npy_path = os.path.join(flip_pred_npy_dir, 'flip_'+img_name+'.jpg.npy')


            # read data
            pred_npy = np.load(pred_npy_path)
            flip_pred_npy = np.load(flip_pred_npy_path)
            gt_img = cv2.imread(os.path.join(gt_mask_dir, gt_img_path))
            gt_img = cv2.resize(gt_img, (input_w, input_h))

            # reshape gt img
            gt_mask = gt_img[: ,:, 0] / 255.0

            # binarize mask
            if use_flip:
                prob_mask = (pred_npy + np.fliplr(flip_pred_npy)) / 2.0
                mask = (prob_mask > thresh).astype(np.uint8)
                dice_sum += dice(mask, gt_mask)
            else:
                prob_mask = pred_npy
                mask = (prob_mask > thresh).astype(np.uint8)
                dice_sum += dice(mask, gt_mask)
        mean_dice = dice_sum / 5088.0

        # try to get lock
        if mutex.acquire(1):
            if mean_dice > best_score['dice']:
                best_score['dice'] = mean_dice
                best_score['thresh'] = thresh
                best_score['use_flip'] = use_flip
            print('{} mean dice = {}, best mean dice = {}'.format(thread_name, mean_dice, best_score))
            mutex.release()




if __name__ == '__main__':

    thread1 = threading.Thread(target=search_thresh, name='thread1', args=['thread1'])
    thread2 = threading.Thread(target=search_thresh, name='thread2', args=['thread2'])
    thread3 = threading.Thread(target=search_thresh, name='thread3', args=['thread3'])
    thread4 = threading.Thread(target=search_thresh, name='thread4', args=['thread4'])

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
