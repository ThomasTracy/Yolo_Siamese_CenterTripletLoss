# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3
from siamesenet.model import SiameseNet
from siamesenet.data_loader import dataloader

from pprint import pprint

tf.reset_default_graph()

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("--input_video", type=str,
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="/home/tracy/YOLOv3_TensorFlow/data/my_data/traffic_sign.names",
                    help="The path of the class names with types of traffic signs.")
parser.add_argument("--class_name_path_all", type=str, default="/home/tracy/YOLOv3_TensorFlow/data/my_data/traffic_sign_all.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="/home/tracy/YOLOv3_TensorFlow/checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.classes_all = read_class_names(args.class_name_path_all)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)

graph_yolo = tf.Graph()
graph_siam = tf.Graph()

sess_yolo = tf.Session(graph=graph_yolo, config=config)
sess_siam = tf.Session(graph=graph_siam)

with sess_yolo.as_default():
    with graph_yolo.as_default():

        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        # Part of Siamese net
        input1 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input1')
        input2 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input2')
        with tf.variable_scope('siamese'):
            label_pred, score, distance = yolo_model.siamese_forward(input1, input2)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.45)

        variables = tf.contrib.framework.get_variables_to_restore()
        variabels_to_restore = [v for v in variables if v.name.split('/')[0] == 'yolov3']
        saver = tf.train.Saver(variabels_to_restore)
        # saver = tf.train.import_meta_graph('./checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05.meta')
        saver.restore(sess_yolo, args.restore_path)

        # Saver of siamese net

        variabels_to_restore_siamese = [v for v in variables if v.name.split('/')[0] == 'siamese']
        # pprint(variabels_to_restore_siamese)
        saver_siamese = tf.train.Saver(variabels_to_restore_siamese)
        saver_siamese.restore(sess_yolo, '/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted/model_alterd')
        # saver_siamese.save(sess_yolo, '/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted/model_after_loaded')

with sess_siam.as_default():
    with graph_siam.as_default():
        siamese_model = SiameseNet()
        siamese_model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/best/my_model')


img = cv2.imread('/home/tracy/YOLOv3_TensorFlow/temp/6_0.jpg')
with sess_yolo.as_default():
    with sess_yolo.graph.as_default():
        img1, img2 = dataloader(img)

        # input1_ = sess_yolo.run(img1)
        # input2_ = sess_yolo.run(img2)
        # label_pred_plh, label_score_plh, distance_plh = sess_yolo.run([label_pred, score, distance], feed_dict={input1:input1_, input2:input2_})


        label_pred_, label_score_, distance_ = yolo_model.siamese_forward(img1, img2)
        print("\033[1;32m**************** result from yolo_siamese net ****************\033[0m")
        sess_yolo.run(tf.global_variables_initializer())
        saver_siamese.restore(sess_yolo, '/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted/model_alterd')

        label = sess_yolo.run(label_pred_)

        # print(label_pred_plh)
        print(label)
sess_yolo.close()

with sess_siam.as_default():
    with sess_siam.graph.as_default():
        img1, img2 = dataloader(img)

        img1_, img2_ = img1.eval(), img2.eval()

        label_pred_siam, label_score_siam, distance_siam = siamese_model.prediction(img1_, img2_)
        label_pred_siam_, label_score_siam_, distance_siam_ = sess_siam.run([label_pred_siam, label_score_siam, distance_siam])


        # intermediate_layer_model = Model(inputs=siamese_model.input,
        #                                  outputs=siamese_model.get_layer(
        #                                      'encoder/layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE').output)
        # intermediate_output = intermediate_layer_model.predict(img1. img2)
        print("\033[1;32m**************** result from siamese net ****************\033[0m")
        # print('layer1: ', intermediate_output)
        print(label_pred_siam_)

        # print('---------------------------------------------------')
        # sess_siam.run(tf.Print(img1, [img1, img1.shape], message='Debug message: '))
        # print('---------------------------------------------------')
        # print(img1_)