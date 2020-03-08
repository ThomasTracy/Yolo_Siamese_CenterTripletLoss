import os
from PIL import Image


def annotation_convert():
    """
    Conver annotations from google API form to Yolo form
    -------

    """
    from_dir = '/home/tracy/data/TrafficSign/Annotations'
    to_dir = '/home/tracy/data/TrafficSign/'
    img_dir = '/home/tracy/data/TrafficSign/Images'
    label_dict = {
        'traffic_sign':0,
        'car':1,
        'pedestrian':2
    }

    anno_list = sorted(os.listdir(from_dir))
    with open(to_dir+'Annotations.txt', 'w') as f_w:
        for i, anno in enumerate(anno_list):
            name = anno.split('.')[0]
            img_path = os.path.join(img_dir, name+'.jpg')
            img = Image.open(img_path)
            width, height = img.size[0], img.size[1]
            read_path = os.path.join(from_dir, anno)
            with open(read_path, 'r') as f_r:
                lines = f_r.readlines()

            f_w.write(str(i) + ' ')
            f_w.write(img_path + ' ')
            f_w.write(str(width) + ' ' + str(height) + ' ')
            for line in lines:
                line = line.strip().split(' ')
                f_w.write(str(label_dict[str(line[4])]) + ' ')
                f_w.write(' '.join(line[0:4]) + ' ')
            f_w.write('\n')


if __name__ == '__main__':
    img_path = '/home/tracy/data/TrafficSign/Images/00000.jpg'
    img = Image.open(img_path)
    size = img.size
    print(size[0], '\t', size[1])
    annotation_convert()