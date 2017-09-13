import numpy as np
import tensorflow as tf
import os
import cv2
from random import shuffle

def train_edge_generator(batch_size):
    img_h = 1280
    img_w = 1918
    train_img_dir = 'D:\Datasets_HDD\Carvana\\train_hq'
    gt_mask_dir = 'D:\Datasets_HDD\Carvana\output_masks'
    model_pred_dir = 'D:\Projects\kaggle-carvana\model_training_predictions\w_contour-360000'
    thresh = 0.3487773093817086

    # for each training img
    while True:
        img_list = os.listdir(train_img_dir)
        shuffle(img_list)

        batch_train_data = []
        batch_gt_mask = []
        current_batch_size = 0
        for img_name in img_list:
            gt_mask_path = os.path.join(gt_mask_dir, img_name.split('.jpg')[0]+'_mask.png')
            model_pred_path = os.path.join(model_pred_dir, img_name+'.npy')
            flip_model_pred_path = os.path.join(model_pred_dir, 'flip_'+img_name+'.npy')

            # read imgs
            train_img = cv2.imread(os.path.join(train_img_dir, img_name))
            gt_mask = cv2.imread(gt_mask_path)
            gt_mask = gt_mask[:, :, 0]
            gt_mask = gt_mask.astype(np.float32)
            gt_mask /= 255.0
            model_pred = np.load(model_pred_path)
            flip_model_pred = np.fliplr(np.load(flip_model_pred_path))
            ensemble_model_pred = (model_pred + flip_model_pred) / 2.0
            model_pred_over_thresh = (ensemble_model_pred > thresh).astype(np.uint8)

            # find contour
            im2, c, hie = cv2.findContours(model_pred_over_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = np.zeros(shape=(1280, 1918, 3))
            cv2.drawContours(contour_img, c, -1, (255, 255, 255), 10)
            contour_img = contour_img / 255.0

            # extract img on contour
            img_on_contour = np.multiply(contour_img, train_img)
            model_pred_on_contour = np.multiply(contour_img[:, :, 0], model_pred)
            model_pred_on_contour = np.expand_dims(model_pred_on_contour, axis=2) * 255.0
            gt_mask_on_contour = np.multiply(contour_img[:, :, 0], gt_mask)

            concat_img = np.concatenate((img_on_contour, model_pred_on_contour), axis=2)

            batch_train_data.append(concat_img)
            batch_gt_mask.append(gt_mask_on_contour)
            current_batch_size += 1

            if current_batch_size == batch_size:
                batch_train = np.array(batch_train_data, np.float32) / 255.0
                batch_train -= 0.5
                batch_gt = np.array(batch_gt_mask, np.float32)
                yield batch_train, batch_gt

                # clear vars
                batch_train_data = []
                batch_gt_mask = []
                current_batch_size = 0



        if current_batch_size != 0:
            print(current_batch_size)
            batch_train = np.array(batch_train_data, np.float32) / 255.0
            batch_train -= 0.5
            batch_gt = np.array(batch_gt_mask, np.float32)
            yield batch_train, batch_gt

            batch_train_data = []
            batch_gt_mask = []
            current_batch_size = 0




                # # diff
                # diff_img = gt_mask - model_pred_over_thresh
                # diff_img_nodif = np.abs(gt_mask - model_pred_over_thresh)
                # diff_pos = (diff_img > 0).astype(np.float32)
                # diff_neg = (diff_img < 0).astype(np.float32)
                # diff_img = np.zeros(shape=(1280, 1918, 3))
                # diff_img[:, :, 0] = diff_pos * 255
                # diff_img[:, :, 1] = diff_neg * 255
                # diff_img = cv2.resize(diff_img, (1024,1024))
                # diff_img_nodif = cv2.resize(diff_img_nodif, (1024,1024))
                #
                #
                # cv2.imshow('t', img_on_contour.astype(np.uint8))
                # cv2.imshow('m', model_pred_on_contour)
                # cv2.imshow('gt', gt_mask_on_contour)
                # cv2.waitKey(0)





if __name__ == '__main__':
    data_gen = train_edge_generator(4)
    while True:
        batch_data, batch_gt = data_gen.__next__()
        print(batch_data.shape, batch_gt.shape)
        batch_data += 0.5
        batch_data *= 255.0
        for i in range(batch_data.shape[0]):
            img = batch_data[i, :, :, 0:3]
            model_pred = batch_data[i, :, :, 3]
            gt_mask = batch_gt[i] * 255.0

            img = cv2.resize(img, (512,512))
            model_pred = np.repeat(np.expand_dims(model_pred, axis=2), 3, axis=2)
            model_pred = cv2.resize(model_pred, (512,512))
            gt_mask = np.repeat(np.expand_dims(gt_mask, axis=2), 3, axis=2)
            gt_mask = cv2.resize(gt_mask, (512,512))
            concat_img = np.concatenate((img, model_pred, gt_mask), axis=1)

            cv2.imshow('i', concat_img.astype(np.uint8))
            cv2.imshow('pred', model_pred)
            cv2.imshow('gt', gt_mask)
            cv2.waitKey(0)



