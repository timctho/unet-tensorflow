import cv2
import numpy as np
import pandas as pd
# from tqdm import tqdm
from model.u_net_tf import UNet
import tensorflow as tf
import os


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



input_size = 1024
batch_size = 1
orig_width = 1918
orig_height = 1280
# threshold = 0.4
model = UNet(input_size)

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
print("Generating submission file...")
df = pd.DataFrame({'threshold': thresh_list, 'score': score_list})
df.to_csv('train_thresh_search.csv')
# df = pd.DataFrame({'img': names, 'rle_mask': rles})
# df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
