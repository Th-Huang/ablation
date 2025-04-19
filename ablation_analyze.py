import os
import cv2
import numpy as np
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
            img = cv2.imread(image_name,2)
            imgs_l.append(img)
        if filename.endswith('1.tif'):
            image_name = os.path.join(image_path, filename)
            img = cv2.imread(image_name,2)
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
    all_data = []
    datas_strain = []
    for filename in data_list:
        if filename.endswith('.csv'):
            data_name = os.path.join(data_path, filename)
            data_df = pd.read_csv(data_name,header=1)
            data = np.array(data_df.values)
            data_strain = data[:,6:9]
            datas_strain.append(data_strain)
            all_data.append(data)
    return all_data,datas_strain
data, data_strain = read_data(data_path)
print("Data length:", len(data))
print("Data strain shape:", data_strain[0].shape)
imge_ldata = []
imge_rdata = []
for i in range(len(data)):
    file = np.array(data[i])
    XY = file[:,13:15]
    img1 = np.array(imgs_l[i])
    img2 = np.array(imgs_r[i])
    img_l = []
    img_r = []
    for j in range(len(XY)):
        datal = [int(XY[j][0]),int(XY[j][1]),int(img1[int(XY[j][0]), int(XY[j][1])])]
        datar = [int(XY[j][0]),int(XY[j][1]),int(img2[int(XY[j][0]), int(XY[j][1])])]
        img_l.append(datal)
        img_r.append(datar)
    img_l = np.array(img_l)
    img_r = np.array(img_r)
    imge_ldata.append(img_l)
    imge_rdata.append(img_r)

imge_rdata = np.array(imge_rdata)
imge_ldata = np.array(imge_ldata)
print("Left data length:", len(imge_ldata))
print("Left data shape:", imge_ldata[0].shape)
print("Right data length:", len(imge_rdata))
print("Right data shape:", imge_rdata[0].shape)