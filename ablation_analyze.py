import os
import cv2

folder_path = 'dataset/image/t1'
imgs = []
image_list = os.listdir(folder_path).sort(key=lambda x: int(x.split('_')[1]))
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image_name = os.path.join(folder_path, filename)
        img = cv2.imread(image_name)
        imgs.append(img)
        print(image_name)
        # break
print(len(imgs))
# # image_path = os.path.join(folder_path, 't1-00000000_0.tif')
# # img = cv2.imread('dataset/image/t1/t1-00000000_0.tif')
#
# print(img.shape)