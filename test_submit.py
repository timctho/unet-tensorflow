import cv2
import numpy as np
import pandas as pd
# from tqdm import tqdm
from model.u_net_tf import UNet
import tensorflow as tf
import os
import time
import numpy as np
import pydensecrf.densecrf as dcrf
from matplotlib import pyplot as plt
from pydensecrf.utils import unary_from_softmax

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

import argparse

parser = argparse.ArgumentParser(description='Process some options.')
parser.add_argument('closing', default=False, type=bool, nargs='?',
                    help='an option for the closing')
parser.add_argument('crf', default=True, type=bool, nargs='?',
                    help='an option for the crf')
args = parser.parse_args()

print("[Option] crf: %r, closing: %r"% (args.crf, args.closing))

def dice(pred_mask, gt_mask):
    intersection = np.sum(np.multiply(pred_mask, gt_mask))
    return (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask))

def CloseInContour( mask, element ):
    large = 0
    result = mask
    _, contours, _ = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, element)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
             pt = cv2.pointPolygonTest(c, (x, y), True)
             #pt = cv2.pointPolygonTest(c, (x, y), False)
             if pt > 3:
                result[x][y] = closing[x][y]
    return result.astype(np.float32)



if args.closing == True:

    def CloseInContour( mask, element ):
        large = 0
        result = mask
        _, contours, _ = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #find the biggest area
        c = max(contours, key = cv2.contourArea)

        closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, element)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                 #pt = cv2.pointPolygonTest(c, (x, y), True)
                 pt = cv2.pointPolygonTest(c, (x, y), False)
                 if pt == 1:
                    result[x][y] = closing[x][y]
        return result.astype(np.float32)


input_size = 1024
batch_size = 1
orig_width = 1918
orig_height = 1280
# threshold = 0.4
model = UNet(input_size)

if args.crf == True:
    label = 2
    d = dcrf.DenseCRF2D(orig_width, orig_height, label)  # width, height, nlabels



# df_test = pd.read_csv('input/sample_submission.csv')
df_test = pd.read_csv('input/train_masks.csv')

ids_test = df_test['img'].map(lambda s: s.split('.')[0])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
saver = tf.train.Saver()

output = model.output_mask
prob_output = tf.sigmoid(output)
sess.run(tf.global_variables_initializer())


model_file = 'w_contour-360000'
if not os.path.exists('model_training_predictions/{}-{}-hq'.format(model_file, input_size)):
    os.mkdir('model_training_predictions/{}-{}-hq'.format(model_file, input_size))
saver.restore(sess, 'model\\{}'.format(model_file))
print('model loaded')

for i in tf.global_variables():
    print(i.name, np.mean(sess.run(i)))

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []

dice_sum = 0
score_list = []
thresh_list = []
highest_score = [0, 0]

# for closing operator
closing_sum = 0
element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for i in range(1):
    thresh = 0.347013
    # thresh = 0.00
    # while thresh < 0.2 or thresh > 0.55:
    #     thresh = np.random.rand()

    thresh_list.append(thresh)
    for start in range(0, len(ids_test), batch_size):
    # for start in range(0, 20, batch_size):
        print(highest_score)
        print(start)
        x_batch = []
        x_flip_batch = []
        mask_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        img_name = ''

        for id in ids_test_batch.values:
            img_name = '{}.jpg'.format(id)
            img = cv2.imread('D:\Datasets_HDD\Carvana\\train_hq\\{}.jpg'.format(id))
            img = cv2.resize(img, (input_size, input_size))
            mask = cv2.imread('D:\Datasets_HDD\Carvana\\output_masks\\{}_mask.png'.format(id), 0)
            mask = mask.astype(np.float32)
            mask /= 255.0
            x_batch.append(img)
            x_flip_batch.append(np.fliplr(img))
            mask_batch.append(mask)
        x_batch = np.array(x_batch, np.float32) / 255
        x_flip_batch = np.array(x_flip_batch, np.float32) / 255
        x_batch -= 0.5
        x_flip_batch -= 0.5

        preds = sess.run([prob_output], feed_dict={model.input: x_batch})
        preds = preds[0]
        preds_flip = sess.run([prob_output], feed_dict={model.input: x_flip_batch})
        preds_flip = preds_flip[0]

        preds = np.squeeze(preds, axis=3)
        preds_flip = np.squeeze(preds_flip, axis=3)
        for i in range(preds.shape[0]):
            gt = mask_batch[i]
            pred = preds[i]
            pred_flip = preds_flip[i]
            prob = cv2.resize(pred, (orig_width, orig_height))
            prob_flip = cv2.resize(pred_flip, (orig_width, orig_height))
            prob_fuse = (prob+np.fliplr(prob_flip)) / 2.0
            mask = (prob_fuse > thresh).astype(np.float32)

            np.save('model_training_predictions\\{}-1280\\{}'.format(model_file, img_name), prob)
            np.save('model_training_predictions\\{}-1280\\flip_{}'.format(model_file, img_name), prob_flip)


            # For closing
            # closing = CloseInContour((mask * 255).astype(np.uint8), element)
            # closing = closing.astype(np.float32)
            # closing = closing / 255
            # closing_sum += dice(closing, gt)
            closing = mask

            print(dice(closing, gt))
            dice_sum += dice(closing, gt)
            # cv2.imshow('m', (mask * 255).astype(np.uint8))
            # cv2.waitKey(0)

            # rle = run_length_encode(closing)
            # rles.append(rle)
    score = dice_sum / 5088.0
    if score > highest_score[1]:
        highest_score[0], highest_score[1] = thresh, score
    score_list.append(score)


print('mean dice: {}'.format(dice_sum / 5088.0))
if args.crf == True:
    crf_sum = 0
# for closing operator
if args.closing == True:
    closing_sum = 0
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) 

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in range(0, len(ids_test), batch_size):
    print(start)
    x_batch = []
    mask_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread('/home/paige/Documents/KAGGLE/IMAGE_SEGMENTATION/unet-tensorflow/input/train/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        mask = cv2.imread('/home/paige/Documents/KAGGLE/IMAGE_SEGMENTATION/unet-tensorflow/input/output_masks/{}_mask.png'.format(id), 0)
        mask = mask.astype(np.float32)
        mask /= 255.0
        x_batch.append(img)

        mask_batch.append(mask)
    x_batch = np.array(x_batch, np.float32) / 255
    x_batch -= 0.5
    # preds = model.predict_on_batch(x_batch)
    preds = sess.run([prob_output], feed_dict={model.input: x_batch})
    preds = preds[0]

    preds = np.squeeze(preds, axis=3)
    for i in range(preds.shape[0]):
        gt = mask_batch[i]
        pred = preds[i]
        prob = cv2.resize(pred, (orig_width, orig_height)) # height, width
        mask = (prob > threshold).astype(np.float32)
        print("dice(mask, gt)")
        print(dice(mask, gt))
        dice_sum += dice(mask, gt)
        # for crf
        if args.crf == True:
            prob = np.transpose(prob,(1,0)) # width, height
            # print(prob.shape)
            # img = np.transpose(x_batch[i], (1,0,2)) # width, height, channel
            img = cv2.resize(img, (orig_width, orig_height))
            U = -np.log([1-prob, prob]) # label, width, height

            U = U.reshape((2,-1)) # Needs to be flat.
            d.setUnaryEnergy(U)
            
            """
            # This potential penalizes small pieces of segmentation that are
            # spatially isolated -- enforces more spatially consistent segmentations
            feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])

            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features --
            # because the segmentation that we get from CNN are too coarse
            # and we can use local color features to refine them
            feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                               img=img.astype(np.uint8), chdim=2)

            d.addPairwiseEnergy(feats, compat=10,
                                 kernel=dcrf.DIAG_KERNEL,
                                 normalization=dcrf.NORMALIZE_SYMMETRIC)
            """
              
            # This adds the color-independent term, features are the locations only.        
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

            #print("x_batch.shape")
            #print(x_batch[i].astype(np.uint8).shape)
            # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
            d.addPairwiseBilateral(sxy=(10, 10), srgb=(3, 3, 3), rgbim=img.astype(np.uint8),
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q = d.inference(5)
            res = np.argmax(Q, axis=0).reshape((orig_width,orig_height))
            # proba = np.array(Q)

            res = res.astype(np.uint8)


            crf = np.transpose(res, (1,0))

            print("dice(crf,gt)")
            print(dice(crf,gt))

            crf_sum += dice(crf, gt)
            res = np.transpose(res,(1,0)) # width, height
            # cv2.imshow('crf', res*255)
            # cv2.waitKey(0)
        
        # for closing
        if args.closing == True:
            
            tStart = time.time()
            closing = CloseInContour((mask * 255).astype(np.uint8), element)
            closing = closing.astype(np.float32)
            closing = closing / 255
            closing_sum += dice(closing, gt)
            print("dice(closing, gt)")
            print(dice(closing, gt))
            print("time: ")
            print(time.time()-tStart)
            
            #cv2.imshow('m', closing)
            #cv2.waitKey(0)
       
        rle = run_length_encode(mask)
        rles.append(rle)
print('mean dice: {}'.format( dice_sum / 5088))
print('mean crf dice: {}'.format( crf_sum / 5088))
print('mean closing dice: {}'.format( closing_sum / 5088))
print("Generating submission file...")
df = pd.DataFrame({'threshold': thresh_list, 'score': score_list})
df.to_csv('train_thresh_search.csv')
# df = pd.DataFrame({'img': names, 'rle_mask': rles})
# df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
