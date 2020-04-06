# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

from center_triplet_loss.train_utils import Params
from center_triplet_loss.input_fn import test_input_fn
from center_triplet_loss.triplet_loss import batch_all_center_triplet_loss

from siamesenet.model import SiameseNet
# from siamesenet.data_loader import dataloader

from tqdm import tqdm

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/traffic_sign.names",
                    help="The path of the class names.")
parser.add_argument("--class_name_path_all", type=str, default="D:/Pycharm/Projects/YOLOv3_TensorFlow/data/my_data/traffic_sign_all.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05",
                    help="The path of the weights to restore.")
parser.add_argument("--output_path", type=str, default='./data/test_result.txt',
                    help="the path of txt which save the detections results")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.classes_all = read_class_names(args.class_name_path_all)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)

tf.reset_default_graph()
graph_yolo = tf.Graph()
graph_triplet = tf.Graph()

sess_yolo = tf.Session(graph=graph_yolo, config=config)
sess_triplet = tf.Session(graph=graph_triplet, config=config)

with sess_yolo.as_default():
    with graph_yolo.as_default():

        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.20)

        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('./checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05.meta')
        saver.restore(sess_yolo, args.restore_path)


params = Params('D:/Pycharm/Projects/YOLOv3_TensorFlow/center_triplet_loss/model/parameters.json')
# put the downloaded frozen graph to path\to
frozen_graph_path = 'path\\to\\frozen_inference_graph.pb'
with sess_triplet.as_default():
    with graph_triplet.as_default():
        od_graph_def = tf.GraphDef()
        # image_input = test_input_fn(image, params)
        with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
            serialized_graph = f.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        inputs = graph_triplet.get_tensor_by_name('image_input:0')
        predict_labels = graph_triplet.get_tensor_by_name('classes:0')

        # distance = batch_all_center_triplet_loss(params, outputs)
        # predict_labels = tf.argmin(distance, axis=1)


def test_one_img(img_path):
    img_ori = cv2.imread(img_path)

    img_name = img_path.strip().split('\\')[-1]
    img_name = img_name.split('.')[0]

    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    with sess_yolo.as_default():
        with graph_yolo.as_default():
            boxes_, scores_, labels_ = sess_yolo.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


    for j in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[j]
        x0 = np.maximum(x0, 0)
        y0 = np.maximum(y0, 0)
        x1 = np.maximum(x1, 0)
        y1 = np.maximum(y1, 0)

        label_index = labels_[j]
        # Crop the detected traffic signs
        # the bbox of traffic signs must be big enough
        if x1 - x0 > 10 and y1 - y0 > 10 and labels_[j] == 0:
            # img_ori_ = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_cropped = img_ori[int(y0):int(y1), int(x0):int(x1)]

            if img_cropped.shape[0]<10 or img_cropped.shape[1]<10:
                continue

            img_cropped = cv2.resize(img_cropped, (params.image_size, params.image_size))
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            img_cropped = img_cropped / 255.0
            # cv2.imwrite('D:\\Data\\TrafficSigns\\test_images\\traffic_sign_cropped\\{}_{}.jpg'.format(img_name, j), img_cropped*255.0)
            # img_cropped_path = 'D:\\Data\\TrafficSigns\\test_images\\traffic_sign_cropped\\{}_{}.jpg'.format(img_name, j)
            # print(img_cropped_path)
            # cv2.imwrite('D:\\Data\\TrafficSigns\\test_images\\traffic_sign_cropped\\1_0.jpg', img_cropped*255.0)

            if img_cropped.any():
                # tf.reset_default_graph()
                # new_graph = tf.Graph()
                with graph_triplet.as_default():
                    with sess_triplet.as_default():
                        image_input = test_input_fn(img_cropped, params)
                        image_input = sess_triplet.run(image_input)
                        label_index = sess_triplet.run(predict_labels, feed_dict={inputs: image_input})
                        label_index = label_index[0] + 3

            # with open('D:/Data/test_result/detect_result_self_collect.txt', 'a+') as f:
            #     f.write(img_path + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
            #         label_index[0]+2) + '\n')
        if isinstance(label_index, np.ndarray):
            label_index = label_index[0]
        with open('D:\Data\TrafficSigns\\test_images/detect_result.txt', 'a+') as f:
            f.write(img_path+' '+str(x0)+' '+str(y0)+' '+str(x1)+' '+str(y1)+' '+str(label_index) + '\n')


def test_display_one_img(img_path):
    print(img_path)
    img_ori = cv2.imread(img_path)
    print(img_ori.shape)
    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    with sess_yolo.as_default():
        with graph_yolo.as_default():
            boxes_, scores_, labels_ = sess_yolo.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


    for j in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[j]
        x0 = np.maximum(x0, 0)
        y0 = np.maximum(y0, 0)
        x1 = np.maximum(x1, 0)
        y1 = np.maximum(y1, 0)

        label_index = labels_[j]
        # Crop the detected traffic signs

        if x1-x0>10 and y1-y0>10 and labels_[j] == 0:
            # img_ori_ = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_cropped = img_ori[int(y0):int(y1), int(x0):int(x1)]

            if img_cropped.shape[0]<10 or img_cropped.shape[1]<10:
                continue

            # cv2.imwrite('D:/Data/TrafficSigns/test/test_{}.png'.format(j), img_cropped)
            img_cropped = cv2.resize(img_cropped, (params.image_size, params.image_size))
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            img_cropped = img_cropped/255.0
            # print(img_cropped)
            # np.savetxt('D:/Data/test_result/img.txt', img_cropped, fmt='%f', delimiter=',')

            if img_cropped.any():
                # tf.reset_default_graph()
                # new_graph = tf.Graph()
                with graph_triplet.as_default():
                    with sess_triplet.as_default():
                        image_input = test_input_fn(img_cropped, params)
                        image_input = sess_triplet.run(image_input)
                        label_index = sess_triplet.run(predict_labels, feed_dict={inputs: image_input})
                        label_index = label_index[0] + 3
                        print(label_index)
                        # with open('D:/Data/test_result/outputs.txt', 'w') as ff:
                        #     ff.writelines(ff)
                # np.savetxt('D:/Data/test_result/outputs.txt', out, fmt='%f', delimiter=',')


        plot_one_box(img_ori, [x0, y0, x1, y1],
                     label_index=label_index,
                     label=args.classes_all[label_index] + ', {:.2f}%'.format(scores_[j] * 100),
                     color=color_table[labels_[j]])

    cv2.namedWindow('Detection result', 0)
    cv2.resizeWindow('Detection result', 2400, 1800)
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)

if __name__ == '__main__':

    dir_list = os.listdir('D:\\Data\\TrafficSigns\\test_images\\Images1')
    dir_list.sort(key=lambda x: int(x[:-4]))
    for img in tqdm(dir_list):
        print("Writting %s"%img)
        test_one_img(os.path.join('D:\\Data\\TrafficSigns\\test_images\\Images1', img))
        print('Done writing %s'%img)

    # # print(os.listdir('/home/tracy/data/TrafficSign_test/Images1'))
    # test_display_one_img('D:\Data\TrafficSigns\\test_images\Images1\\149.jpg')
    # # test_one_img('/home/tracy/data/TrafficSign_test/Images1/8.jpg')
