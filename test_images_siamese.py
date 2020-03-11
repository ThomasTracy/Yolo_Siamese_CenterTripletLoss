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

from siamesenet.model import SiameseNet
from siamesenet.data_loader import dataloader

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
parser.add_argument("--class_name_path_all", type=str, default="/home/tracy/YOLOv3_TensorFlow/data/my_data/traffic_sign_all.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.classes_all = read_class_names(args.class_name_path_all)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

graph_yolo = tf.Graph()
graph_siam = tf.Graph()

sess_yolo = tf.Session(graph=graph_yolo, config=config)
sess_siam = tf.Session(graph=graph_siam, config=config)

with sess_yolo.as_default():
    with graph_yolo.as_default():

        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.40)

        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('./checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05.meta')
        saver.restore(sess_yolo, args.restore_path)

with sess_siam.as_default():
    with graph_siam.as_default():
        siamese_model = SiameseNet()
        siamese_model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/my_model')


def test_one_img(img_path):
    img_ori = cv2.imread(img_path)
    # print(img_path)
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
        if x1 - x0 > 10 and y1 - y0 > 10 and labels_[j] == 0:
            img_ori_ = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_cropped = img_ori_[int(y0):int(y1), int(x0):int(x1)]
            if img_cropped.any():
                tf.reset_default_graph()
                new_graph = tf.Graph()
                with new_graph.as_default():
                    with tf.Session(graph=new_graph) as new_sess:
                        siamese_model = SiameseNet()
                        siamese_model.load_weights(
                            '/home/tracy/PycharmProjects/SiameseNet/checkpoint/RGBscaled/best/my_model')
                        img1, img2 = dataloader(img_cropped)
                        label_pred, label_score, _ = siamese_model.prediction(img1, img2)
                        label_pred_, label_score_ = new_sess.run([label_pred, label_score])

                # with sess_siam.as_default():
                #     with sess_siam.graph.as_default():
                #         img1, img2 = dataloader(img_cropped)
                #         label_pred, label_score, _ = siamese_model.prediction(img1, img2)
                #         label_pred_, label_score_ = sess_siam.run([label_pred, label_score])

                # cv2.imwrite('/home/tracy/YOLOv3_TensorFlow/temp/' + str(i) + '_' + str(j) + '.jpg', img_cropped)

    #     print("Writting %s"%img)
    #     test_one_img('/home/tracy/data/TrafficSign_test/Images1/' + img)
    #     print('Done writing %s'%img)
                # Choose the one label with highest score
                pred_labels = np.nonzero(label_pred_)
                pred_scores = label_score_[pred_labels]
                # print("pred_scores: ", pred_scores)
                if len(pred_scores) > 0:
                    label_index = np.argmax(pred_scores)
                    label_index = pred_labels[0][label_index] + 2
                # labels_[j] = label_index

        with open('/home/tracy/YOLOv3_TensorFlow/detect_result_self_collect.txt', 'a+') as f:
            f.write(img_path+' '+str(x0)+' '+str(y0)+' '+str(x1)+' '+str(y1)+' '+str(label_index) + '\n')


def test_display_one_img(img_path):
    img_ori = cv2.imread(img_path)
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

        if x1 - x0 > 10 and y1 - y0 > 10 and labels_[j] == 0:
            img_ori_ = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_cropped = img_ori_[int(y0):int(y1), int(x0):int(x1)]

            if img_cropped.any():
                tf.reset_default_graph()
                new_graph = tf.Graph()
                with new_graph.as_default():
                    with tf.Session(graph=new_graph) as new_sess:
                        siamese_model = SiameseNet()
                        siamese_model.load_weights(
                            '/home/tracy/PycharmProjects/SiameseNet/checkpoint/RGBscaled/best/my_model')
                        img1, img2 = dataloader(img_cropped)
                        label_pred, label_score, _ = siamese_model.prediction(img1, img2)
                        label_pred_, label_score_ = new_sess.run([label_pred, label_score])

                # with sess_siam.as_default():
                #     with sess_siam.graph.as_default():
                #         img1, img2 = dataloader(img_cropped)
                #         label_pred, label_score, _ = siamese_model.prediction(img1, img2)
                #         label_pred_, label_score_ = sess_siam.run([label_pred, label_score])

                # cv2.imwrite('/home/tracy/YOLOv3_TensorFlow/temp/' + str(i) + '_' + str(j) + '.jpg', img_cropped)

    #     print("Writting %s"%img)
    #     test_one_img('/home/tracy/data/TrafficSign_test/Images1/' + img)
    #     print('Done writing %s'%img)
                # Choose the one label with highest score
                pred_labels = np.nonzero(label_pred_)
                pred_scores = label_score_[pred_labels]
                # print("pred_scores: ", pred_scores)
                if len(pred_scores) > 0:
                    label_index = np.argmax(pred_scores)
                    label_index = pred_labels[0][label_index] + 2
                # labels_[j] = label_index


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

    dir_list = os.listdir('/home/tracy/data/self_collected')
    dir_list.sort(key=lambda x: int(x[:-4]))
    for img in tqdm(dir_list):
        print("Writting %s"%img)
        test_one_img('/home/tracy/data/self_collected/' + img)
        print('Done writing %s'%img)

    # print(os.listdir('/home/tracy/data/TrafficSign_test/Images1'))
    # test_display_one_img('/home/tracy/data/self_collected/175.JPG')
    # test_one_img('/home/tracy/data/TrafficSign_test/Images1/8.jpg')
