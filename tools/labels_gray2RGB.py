# TODO:Convert labels in RGB format to 8-bit grayscale format
import glob
import json
import os
import cv2
import numpy as np

# args
input_path = r"D:/python_codes/MmSAM/dataset/whu-opt-sar/test/gray_label/*png"
save_path = r"D:/python_codes/MmSAM/dataset/whu-opt-sar/test/rgb_label/"
json_label_path = "../json/whuclasses.json"

assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
color = json.load(json_file)


def gray2rgb(input_img, save_img):
    img = cv2.imread(input_img, 0)
    h, w = img.shape
    rgb_img = np.zeros([h, w, 3], dtype=int)
    for i, v in enumerate(color.values()):
        idx = np.where(img == i)
        rgb_img[idx] = v
    r, g, b = cv2.split(rgb_img)
    image = cv2.merge([b, g, r])
    cv2.imwrite(save_img + input_img.split('\\')[-1].split('.')[0] + '.jpg', image)


labels = glob.glob(input_path)
for label in labels:
    gray2rgb(label, save_path)
    print('{} conversion completed!'.format(label.split('\\')[-1]))
print('\nover')
