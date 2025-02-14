# TODO:Transfer background class to last index
from glob import glob
import numpy as np
import cv2

in_path = r"D:/python_codes/MmSAM/dataset/whu-opt-sar/test/crop/0gray_label/*png"
save_path = r"D:/python_codes/MmSAM/dataset/whu-opt-sar/test/crop/gray_label/"

for i in glob(in_path):
    name = i.split('\\')[-1]
    img = cv2.imread(i, -1)
    idx = np.where(img == 0)
    img[idx] = 8
    img -= 1
    cv2.imwrite(save_path + name, img)

print('over')
