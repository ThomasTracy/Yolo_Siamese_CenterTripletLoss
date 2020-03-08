from siamesenet.model import SiameseNet
# from data.data_loader import DataSet
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import numpy as np
from siamesenet import data_loader


def show_result(img1, img2, label):
    num = img1.get_shape()[0]

    for i in range(num):
        ax1 = plt.subplot(5,4,2*i + 1)
        ax1.imshow(img1[i].numpy())
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(label[i].numpy())

        ax2 = plt.subplot(5,4,2*i+2)
        ax1.set_title(label[i].numpy())
        ax2.imshow(img2[i].numpy())
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.show()


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        model = SiameseNet()

        model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/my_model')

        img = cv2.imread('/home/tracy/YOLOv3_TensorFlow/temp/2_3.jpg')

        img1, img2 = data_loader.dataloader(img)
        pred, scores, _ = model.prediction(img1, img2)

        pred, scores = sess.run([pred, scores])

        print(pred, scores)
        pred_labels = np.nonzero(pred)
        scores_ = scores[pred_labels]

        label_index = np.argmax(scores_)

        print('\033[1;32m   Label\t\t\tScore\033[0m')
        for i in pred_labels[0]:
            print('\t', i, '\t\t\t', scores[i], '\n')

        # print()

def yolo_test():
    pass

# img1, img2 = next(test_dataset)
# label = model.prediction(img1, img2)
# print(label)
#
# show_result(img1, img2, label)
if __name__ == '__main__':
    main()