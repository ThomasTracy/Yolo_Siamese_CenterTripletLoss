import cv2
import os
import pprint
import argparse
import numpy as np

from utils.misc_utils import read_class_names
from utils.plot_utils import get_color_table, plot_one_box


parser = argparse.ArgumentParser(description="Paser for display")

parser.add_argument("--class_name_path", type=str, default="./data/my_data/traffic_sign.names",
                    help="The path of the class names.")
parser.add_argument("--class_name_path_all", type=str, default="./data/my_data/traffic_sign_all.names",
                    help="The path of the class names.")
parser.add_argument("--output_path", type=str, default='./data/test_result.txt',
                    help="the path of txt which save the detections results")
args = parser.parse_args()

args.classes = read_class_names(args.class_name_path)
args.classes_all = read_class_names(args.class_name_path_all)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)


def display_results():
    result_txt = output_path
    with open(result_txt, 'r') as f:
        lines = f.readlines()
        lines.sort(key=lambda x: int(x.split(' ')[0].split('/')[-1][:-4]))

    # for i in lines:
    #     print(i.strip().split(' ')[0])

    img_path_last = lines[0].strip().split(' ')[0]
    # print(img_path_last)
    img_last = cv2.imread(img_path_last)
    for i in range(len(lines)):

        line = lines[i].strip().split(' ')
        img_path = line[0]
        x0, y0, x1, y1 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        label_index = int(line[5])
        color_index = lambda x: x if x < 3 else 0

        img = cv2.imread(img_path)
        if img_path != img_path_last:
            cv2.namedWindow('Result', 0)
            cv2.resizeWindow('Result', 1800, 1200)
            cv2.imshow('Result', img_last)
            cv2.waitKey(delay=500)
            img_path_last = img_path
            img_last = img

        plot_one_box(img_last, [x0, y0, x1, y1],
                     label_index=label_index,
                     label=args.classes_all[label_index],
                     color=color_table[color_index(label_index)])


if __name__ == '__main__':
    display_results()
