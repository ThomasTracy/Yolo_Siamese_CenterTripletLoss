import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy
import os

from center_triplet_loss.train_utils import Params


IMG_SIZE = 64

def load_traffic_signs(txt_path):
    """
    Load path and labels of all traffic signs
    from a txt file
    """
    images = []
    labels = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        images.append(l.strip().split(',')[0])
        labels.append(int(l.strip().split(',')[1]))

    return images, labels

def load_img(img_path):
    img = tf.io.read_file(img_path)

    # Turn image in 64x64x3
    img = tf.image.decode_jpeg(img, channels=3)

    # Convert from (0,255) to (0,1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

def normalize(img):

    # Standarize: (image - mean)/std
    # Standard mean and std from ImageNet
    img_mean = tf.constant([0.485, 0.456, 0.406])
    img_std = tf.constant([0.229, 0.224, 0.225])

    img_ = (img - img_mean)/img_std
    return img_

def preprocess(image, label):


    # labels = tf.cast(labels, tf.int32)
    # labels = [int(l) for l in labels]
    img = load_img(image)

    return img, label


def build_ref_dataset(ref_dir):
    images = []
    labels = []
    ref_list =  os.listdir(ref_dir)
    ref_list.sort(key=lambda x: int(x.split('.')[0]))
    for ref in ref_list:
        if ref.endswith('.jpg'):
            label = ref.strip().split('.')[0]
            labels.append(int(label))
            image = os.path.join(ref_dir, ref)
            images.append(image)

    data_size = len(labels)
    dataset_ref = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset_ref = dataset_ref.shuffle(data_size)
    dataset_ref = dataset_ref.map(preprocess)
    # hear repeat is important, otherwise it will end in 1 step
    dataset_ref = dataset_ref.repeat(99999)
    dataset_ref = dataset_ref.batch(data_size)

    return dataset_ref


def train_input_fn(params):
    # Return is not Dataset class
    # instead is image, label
    # the last 51-dims of batch-dim are references
    images, labels = load_traffic_signs(params.train_txt)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(params.train_size, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(params.batch_size, drop_remainder=True) # Make sure all batch are 64, divided evenly
    dataset = dataset.repeat(3)
    dataset = dataset.prefetch(1)

    dataset_ref = build_ref_dataset(params.references_dir)

    iterator = dataset.make_one_shot_iterator()
    iterator_ref = dataset_ref.make_one_shot_iterator()
    imgs, lbls = iterator.get_next()
    img_ref, lbl_ref = iterator_ref.get_next()
    imgs = tf.concat([imgs, img_ref], axis=0)
    lbls = tf.concat([lbls, lbl_ref], axis=0)

    return imgs, lbls


def test_input_fn(image, params):
    # Input test image is a single cv2 image
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    dataset_ref = build_ref_dataset(params.references_dir)
    iterator_ref = dataset_ref.make_one_shot_iterator()
    img_ref, _ = iterator_ref.get_next()

    input_images = tf.concat([image, img_ref], axis=0)

    return input_images





def show_dataset(img, label):
    """
    Use class Iterator to get each
    elements in Dataset
    """
    # iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        img, label = sess.run([img, label])

    for i in range(115):
        ax =plt.subplot(10,12,i+1)
        ax.imshow(img[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label[i])

    plt.show()
    print(img.shape)


if __name__ == '__main__':
    params = Params('../model/parameters.json')
    imgs, labels = build_dataset(params)

    show_dataset(imgs, labels)


    # iterator = dataset.make_one_shot_iterator()
    # img, label = iterator.get_next()
    # writer = tf.summary.FileWriter('../logging/tensorboard')
    #
    # with tf.Session() as sess:
    #     summary = sess.run(tf.summary.image('img', img))
    #     writer.add_summary(summary)