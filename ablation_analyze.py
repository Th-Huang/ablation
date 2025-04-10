import glob
import os
import cv2
import pandas as pd

image_path = 'dataset/image/t1'
data_path = 'dataset/data/t1'

def read_image(image_path):
    imgs_l = []
    imgs_r = []
    image_list = os.listdir(image_path)
    image_list.sort(key=lambda x: int(x.replace('t1-', '').split('_')[0]))
    for filename in image_list:
        if filename.endswith('0.tif'):
            image_name = os.path.join(image_path, filename)
            img = cv2.imread(image_name)
            imgs_l.append(img)
        if filename.endswith('1.tif'):
            image_name = os.path.join(image_path, filename)
            img = cv2.imread(image_name)
            imgs_r.append(img)
    return imgs_l, imgs_r




imgs_l, imgs_r = read_image(image_path)
print("Left images:", len(imgs_l))
print("Right images:", len(imgs_r))

def read_data(data_path):
    data_list = os.listdir(data_path)
    data_list.sort(key=lambda x: int(x.replace('t1_', '').split('_')[0]))
    data = []
    for filename in data_list:
        if filename.endswith('.csv'):
            data_name = os.path.join(data_path, filename)
            with open(data_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    data.append(line)
    return data
data = read_data(data_path)
print("Data length:", len(data))
print(data)