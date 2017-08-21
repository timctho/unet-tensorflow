import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from model.u_net_tf import UNet
from tqdm import tqdm

input_size = 1024
batch_size = 1
orig_width = 1918
orig_height = 1280
threshold = 0.5

model = UNet(1024)

df_test = pd.read_csv('input\\sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

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

graph = tf.get_default_graph()
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'model\\-69999')

output = model.output_mask
prob_output = tf.sigmoid(output)

q_size = 10


def data_loader(q, ):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            img = cv2.imread('D:\Datasets_HDD\Carvana\\test\\{}.jpg'.format(id))
            img = cv2.resize(img, (input_size, input_size))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, ):
    for i in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds = sess.run([prob_output], feed_dict={model.input: x_batch})
            preds = preds[0]

            # preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for i in range(preds.shape[0]):
            pred = preds[i]
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > threshold
            cv2.imshow('m', (mask*255).astype(np.uint8))
            cv2.waitKey(0)
            rle = run_length_encode(mask)
            rles.append(rle)


q = queue.Queue(maxsize=q_size)
t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
t1.start()
t2.start()
# Wait for both threads to finish
t1.join()
t2.join()

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit\\submission.csv.gz', index=False, compression='gzip')
