import os
import tensorflow as tf
import numpy as np
import cv2


REFERENCES_DIR='/home/tracy/data/traffic_sign_ref'
IMG_LIST = os.listdir(REFERENCES_DIR)
IMG_LIST.sort(key=lambda x: int(x[:-4]))
REFERENCES_LIST = list(map(lambda x: REFERENCES_DIR+'/'+x,
                           IMG_LIST))
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def load_img(img_path):
    img = tf.io.read_file(img_path)

    # Turn image in 64x64x3   0 -- 1.0
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.rgb_to_grayscale(img)
    return tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])
    # img = tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])
    # return tf.expand_dims(img, axis=0)


def dataloader(input_img):
    # img1 = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img1 = input_img
    img1 = cv2.normalize(input_img, img1, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
    img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    # img1 = tf.expand_dims(img1, axis=-1)      # 当是色彩图的时候就不用最后扩张一个维度
    img1 = tf.image.resize(img1, [IMAGE_WIDTH, IMAGE_HEIGHT])

    img1 = tf.expand_dims(img1, axis=0)
    img1 = tf.tile(img1, [len(REFERENCES_LIST), 1, 1, 1])
    img2 = list(map(load_img, REFERENCES_LIST))

    # print(img1[0].shape, img2[0].shape)

    img1 = tf.concat(img1, axis=0)
    # img2 = tf.concat(img2, axis=0)
    img2 = tf.stack(img2, axis=0)

    return img1, img2

def dataloader_np(input_img):
    img1 = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img1 = cv2.normalize(img1, img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    img1 = np.expand_dims(img1, axis=-1)
    img1 = tf.image.resize(img1, [IMAGE_WIDTH, IMAGE_HEIGHT])

    img1 = tf.expand_dims(img1, axis=0)
    img1 = tf.tile(img1, [len(REFERENCES_LIST), 1, 1, 1])
    img2 = list(map(load_img, REFERENCES_LIST))

    img1 = tf.concat(img1, axis=0)
    # img2 = tf.concat(img2, axis=0)
    img2 = tf.stack(img2, axis=0)

    return img1, img2


if __name__ == '__main__':
    img = cv2.imread('/home/tracy/data/TrafficSign_single/train/00003/00000_00029.jpg')
    i1, i2 = dataloader(img)
    with tf.Session() as sess:
        img1 = sess.run(i1[0])
    cv2.imshow('result', img1)
    cv2.waitKey(0)