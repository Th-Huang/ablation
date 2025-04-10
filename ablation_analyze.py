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
print("Left images:", imgs_l[0].shape)
print("Right images:", len(imgs_r))
print("Right images:", imgs_r[0].shape)

def read_data(data_path):
    data_list = os.listdir(data_path)
    data_list.sort(key=lambda x: int(x.replace('t1_', '').split('_')[0]))
    datas = []
    for filename in data_list:
        if filename.endswith('.csv'):
            data_name = os.path.join(data_path, filename)
            data_df = pd.read_csv(data_name,header=1,index_col=0)
            data = data_df.values
            datas.append(data)
    return datas
data = read_data(data_path)
print("Data length:", len(data))
for i in range(len(data)):
    print(data[i].shape)