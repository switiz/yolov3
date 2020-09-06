
Tag = 'coco_yolo_txt'

"""colab check"""
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    pass
else:
    pass

import json
import os
from collections import defaultdict

from pycocotools.coco import COCO
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def check_dir(path):
    if path == 'colab':
        images_dir_path = '/content/drive/' + 'Shared drives' + '/YS_NW/2.Data/Train/Data'
        json_file_path = '/content/drive/Shared drives/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'google_drive':
        images_dir_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        json_file_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'local_d':
        images_dir_path = 'D:/Local/Train/Data'
        json_file_path = 'D:/Local/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'local_c':
            images_dir_path = 'C:/Local/Train/Data'
            json_file_path = 'D:/Local/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'local_e':
        images_dir_path = 'E:/dataset/xray/Train/Data'
        json_file_path = 'E:/dataset/xray/Train/Meta/CoCo/coco_rapiscan.json'

    return images_dir_path, json_file_path


def select_class(path, classes):
    if classes in path:
            return True
    else:
        return False

def convert_labels(input):
    height, width = 1050, 1680
    x1, y1, x2, y2 = input[0], input[1], input[2], input[3]
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    xmin = x1
    xmax = x1 + x2
    ymin = y1
    ymax = y1 + y2
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = float(x / width)
    w = float(w / width)
    y = float(y / height)
    h = float(h / height)
    # print('input : x1:',x1,',y1:',y1,',x2:',x2,',y2:',y2)
    # print('output : x:',x,',y:',y,',w:',w,',h:',h)
    return (x, y, w, h)

def index_change(index):
    # yolo index start 0 and index 14, 35 is empty
    if index <=13:
        index-=1
    elif index >14 and index <=34:
        index-=2
    elif index >35:
        index-=3
    return int(index)

def from_yolo_to_cor(box):
    height, width = 1050, 1680
    x,y,w,h = box[0], box[1], box[2], box[3]
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x = x*width
    w = w*width
    y = y*height
    h = h*height
    x1 = int(x-w/2)
    x2 = int(x+w/2 -x1)
    y1 = int(y-h/2)
    y2 = int(y+h/2 - y1)
    return x1, y1, x2, y2

def make_coco_to_yolo(cat_type):
    name_box_id = defaultdict(list)
    for ant in tqdm(annotations):
        id = ant['image_id']
        # name = os.path.join(images_dir_path, images[id]['file_name'])
        ann_ids = coco.getAnnIds(imgIds=id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(coco_annotation[0]["image_id"])
        cat = coco.getCatIds(catIds=id)
        file_path = img_info[0]["path"].split('\\', maxsplit=7)[-1]
        if select_class(file_path, cat_type) is False:
            continue
        if os.path.isfile(os.path.join(images_dir_path, file_path)) is False:
            # log(Tag , 'empty file : '+str(file_path))
            continue
        name = os.path.join(images_dir_path, file_path).replace('\\', '/')
        name_box_id[name].append([convert_labels(ant['bbox']), index_change(ant['category_id'])])

    return name_box_id

def write_anno_file(name_box_id, output, individual):
    """write to txt"""
    if individual:
        for key in tqdm(name_box_id.keys()):
            path = key[:-4]+'.txt'
            with open(path, 'w', encoding='utf-8') as f:
                box_infos = name_box_id[key]
                for idx, info in enumerate(box_infos):
                    x = info[0][0]
                    y = info[0][1]
                    w = info[0][2]
                    h = info[0][3]
                    c = info[1]
                    box_info = "%d %f %f %f %f" % (
                        c, x, y, w, h)
                    f.write(box_info)
                    if idx!=len(box_infos)-1:
                        f.write('\n')
            f.close()
    else :
        with open(output, 'w', encoding='utf-8') as f:
            for key in tqdm(name_box_id.keys()):
                f.write(key)
                box_infos = name_box_id[key]
                for idx, info in enumerate(box_infos):
                    x = info[0][0]
                    y = info[0][1]
                    w = info[0][2]
                    h = info[0][3]
                    c = info[1]
                    box_info = " %f,%f,%f,%f,%d" % (
                        x, y, w, h, c)
                    f.write(box_info)
                    if idx!=len(box_infos)-1:
                        f.write('\n')
        f.close()
def write_path_file(train_data, val_data, output, valout):
    v3_path = './data/custom'
    if os.path.isdir(v3_path) is False:
        os.mkdir(v3_path)
    output = os.path.join(v3_path, output)
    valout = os.path.join(v3_path, valout)

    """write to txt"""
    with open(output, 'w', encoding='utf-8') as f:
        keys = train_data.keys()
        for idx, key in enumerate(keys):
            f.write(key)
            if idx !=len(keys) -1:
                f.write('\n')
    f.close()

    """write to txt"""
    with open(valout, 'w', encoding='utf-8') as f:
        keys = val_data.keys()
        for idx, key in enumerate(keys):
            f.write(key)
            if idx !=len(keys) -1:
                f.write('\n')
    f.close()

"""split tran / val"""
def train_val_split(name_box_id):

    ### add group
    dictlist = []
    for key, value in name_box_id.items():
        temp = [key, value]
        dictlist.append(temp)

    dictlist = np.array(dictlist, dtype=object)

    df = pd.DataFrame(dictlist, columns=['data', 'label'])
    df['grp'] = df['data'].str.split('/').str[5] + "_" + df['data'].str.split('/').str[6] #for local
    # df['grp'] = df['data'].str.split('/').str[5] + "_" + df['data'].str.split('/').str[6] #for google drive

    name_box_id = df.values.tolist()

    name_box_id = np.array(name_box_id, dtype=object)
    grp = name_box_id[:,-1]

    "split by groups (ex: Battery_Single Others)"
    x_train, x_val, y_train, y_val = train_test_split(name_box_id[:,0], name_box_id[:,1], test_size=0.25, stratify=grp)

    train_data = defaultdict(list)
    for i in range(len(x_train)):
        print(i)
        train_data[x_train[i]].append(y_train[i])

    val_data = defaultdict(list)
    for i in range(len(x_val)):
        print(i)
        val_data[x_val[i]].append(y_val[i])

    return train_data, val_data


if __name__ == '__main__':
    """parameters"""
    images_dir_path, json_file_path = check_dir('local_e')
    #    output_paths = ['train_sd.txt', 'train_so.txt', 'train_md.txt', 'train_mo.txt']
    #    cat_types = ['Single_Default', 'Single_Other', 'Multiple_Categories', 'Multiple_Other']
    output_paths = ['train_sd.txt']
    val_paths = ['val_sd.txt']
    cat_types = ['Single_Default']


    """load json file"""
    id_name = dict()
    coco = COCO(json_file_path)
    with open(json_file_path, encoding='utf-8') as f:
        data = json.load(f)

    """generate labels"""
    images = data['images']
    annotations = data['annotations']
    for ouput, valout, cat_type in zip(output_paths, val_paths, cat_types):
        name_box_id = make_coco_to_yolo(cat_type)
        write_anno_file(name_box_id, ouput, True)

        "split train/val"
        train_data, val_data = train_val_split(name_box_id)
        write_path_file(train_data, val_data, ouput, valout)


#check file
# import cv2
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
#
#
# file = 'S_8523.51-1000_01_366.png'
# for idx, key in enumerate(name_box_id.keys()):
#     if file in key:
#         img = Image.open(key)
#         info = name_box_id[key]
#         draw = ImageDraw.Draw(img)
#         data = from_yolo_to_cor(info[0][0])
#         x1,y1,x2,y2 = data[0],data[1],data[2],data[3]
#         draw.rectangle((x1,y1,x2,y2), outline='red', width=3)
#         img.show()