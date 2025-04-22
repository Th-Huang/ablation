import os
import cv2
import numpy as np
import pandas as pd
import torch

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
    datas_strain = np.array(datas_strain)
    return all_data,datas_strain

def available_image(imgs_l, imgs_r,data):
    imge_ldata = []
    imge_rdata = []
    outputstrain = []
    for i in range(len(data)):
        file = np.array(data[i])
        # for j in range(file.shape[0]):
        #     if file[j][0] == 0 and file[j][1] == 0 and file[j][2] == 0:
        #         #delete the row
        #         file = np.delete(file, j, axis=0)
        XY = file[:,13:15]
        X_max = int(np.max(XY[:,0]))
        X_min = int(np.min(XY[:,0]))
        Y_max = int(np.max(XY[:,1]))
        Y_min = int(np.min(XY[:,1]))
        strain = file[:,6:9]
        img1 = np.array(imgs_l[i])
        img2 = np.array(imgs_r[i])

        img1 = img1[X_min:X_max+1, Y_min:Y_max+1]
        img2 = img2[X_min:X_max+1, Y_min:Y_max+1]

        img_l = np.zeros((img1.shape[0], img1.shape[1], 1), dtype=np.uint8)
        img_r = np.zeros((img2.shape[0], img2.shape[1], 1), dtype=np.uint8)
        outstrain = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        for j in range(len(XY)):
            x = int(XY[j][0])
            y = int(XY[j][1])
            if x<X_min or x>X_max or y<Y_min or y > Y_max:
                print("Invalid coordinates:", x, y)
                continue
            img_l[x-X_min][y-Y_min]=int(img1[x-X_min,y-Y_min])
            img_r[x-X_min][y-Y_min]=int(img2[x-X_min,y-Y_min])
            outstrain[x-X_min][y-Y_min][0]=int(strain[j][0])
            outstrain[x-X_min][y-Y_min][1]=int(strain[j][1])
            outstrain[x-X_min][y-Y_min][2]=int(strain[j][2])
        img_l = np.array(img_l)
        img_r = np.array(img_r)
        outstrain = np.array(outstrain)

        imge_ldata.append(img_l)
        imge_rdata.append(img_r)
        outputstrain.append(outstrain)

    imge_rdata = np.array(imge_rdata)
    imge_ldata = np.array(imge_ldata)
    outputstrain = np.array(outputstrain)

    return imge_ldata, imge_rdata,outputstrain

def CustomDataset(imge_ldata, imge_rdata, data_strain):
    # Custom dataset class to handle the data
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, imge_ldata, imge_rdata, data_strain):
            self.imge_ldata = imge_ldata
            self.imge_rdata = imge_rdata
            self.data_strain = data_strain

        def __len__(self):
            return len(self.imge_ldata)

        def __getitem__(self, idx):
            return (self.imge_ldata[idx], self.imge_rdata[idx], self.data_strain[idx])
    return CustomDataset(imge_ldata, imge_rdata, data_strain)


imgs_l, imgs_r = read_image(image_path)
print("Data length:", len(imgs_l))
print("Data strain shape:", imgs_l[0].shape)

data, data_strain = read_data(data_path)
print("Data length:", len(data))
print("Data strain shape:", data_strain[0].shape)
data1 = np.array(data)
print("Data shape:", data1.shape)
X = data1[:,:,13]
Y = data1[:,:,14]

print('X max',np.max(X))
print('X min',np.min(X))
print('Y max',np.max(Y))
print('Y min',np.min(Y))

imge_ldata, imge_rdata, outputstrain = available_image(imgs_l, imgs_r,data)