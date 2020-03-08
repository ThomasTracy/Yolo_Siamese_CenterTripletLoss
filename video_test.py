# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

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
parser.add_argument("--class_name_path", type=str, default="D:/Pycharm/Projects/YOLOv3_TensorFlow/data/my_data/traffic_sign.names",
                    help="The path of the class names with types of traffic signs.")
parser.add_argument("--class_name_path_all", type=str, default="D:/Pycharm/Projects/YOLOv3_TensorFlow/data/my_data/traffic_sign_all.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="D:/Pycharm/Projects/YOLOv3_TensorFlow/checkpoint/model-epoch_490_step_66284_loss_0.3861_lr_1e-05",
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

class SiamesModel():
    def __init__(self, config):
        self.graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.siamese_model = SiameseNet()
                self.siamese_model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/best/my_model')

    def run(self, img1, img2):
        with self.graph.as_default():
            return self.siamese_model.prediction(img1, img2)

# sess_siam = tf.Session(config=config)
# with sess_siam.as_default():
#     siamese_model = SiameseNet()
#     siamese_model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/my_model')

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
            label_pred, score, _ = yolo_model.siamese_forward(input1, input2)

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
        saver_siamese.save(sess_yolo, '/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted/model_after_loaded')

with sess_siam.as_default():
    with graph_siam.as_default():
        siamese_model = SiameseNet()
        siamese_model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/best/my_model')


for i in range(video_frame_cnt):
    print("\033[1;32m**************** frame %d ****************\033[0m"%i)
    ret, img_ori = vid.read()

    # Sometimes the last frame is None
    if img_ori is None:
        continue

    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    start_time = time.time()
    with sess_yolo.as_default():
        with sess_yolo.graph.as_default():
            boxes_, scores_, labels_ = sess_yolo.run([boxes, scores, labels], feed_dict={input_data: img})

    end_time = time.time()

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


    for j in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[j]
        label_index = labels_[j]
        # Crop the detected traffic signs
        if x1 - x0 > 10 and y1 - y0 > 10 and labels_[j] == 0:
            img_ori_ = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_cropped = img_ori_[int(y0):int(y1), int(x0):int(x1)]

            tf.reset_default_graph()
            new_graph = tf.Graph()
            with new_graph.as_default():
                with tf.Session(graph=new_graph) as new_sess:
                    siamese_model = SiameseNet()
                    siamese_model.load_weights(
                        '/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/my_model')
                    img1, img2 = dataloader(img_cropped)
                    label_pred, label_score, _ = siamese_model.prediction(img1, img2)
                    label_pred_, label_score_ = new_sess.run([label_pred, label_score])

            # with sess_yolo.as_default():
            #     with sess_yolo.graph.as_default():
            #         img1, img2 = dataloader(img_cropped)
            #         # input1_ = sess_yolo.run(img1)
            #         # input2_ = sess_yolo.run(img2)
            #         label_pred_, label_score_ = yolo_model.siamese_forward(img1, img2)
            #         print(label_pred_, label_score_)

            # with sess_siam.as_default():
            #     with sess_siam.graph.as_default():
            #         img1, img2 = dataloader(img_cropped)
            #         label_pred, label_score = siamese_model.prediction(img1, img2)
            #         label_pred_, label_score_ = sess_siam.run([label_pred, label_score])
            #         print(label_pred_)

            cv2.imwrite('/home/tracy/YOLOv3_TensorFlow/temp/'+str(i)+'_'+str(j)+'.jpg', img_cropped)

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

    cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow('image', img_ori)
    # cv2.waitKey(delay=300)
    if args.save_video:
        videoWriter.write(img_ori)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
if args.save_video:
    videoWriter.release()