# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 9:38
# @Author  : DaiPuWei
# @File    : count_image_resolution.py
# @Software: PyCharm

import imp
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def count_image_resolution(dataset_dir):
    """
    这是统计数据集图像分辨率分布的函数
    :param dataset_dir: 数据集地址
    :return:
    """
    data = []
    # for label in os.listdir(dataset_dir):
    #     label_dir = os.path.join(dataset_dir,label)
    for image_name in tqdm(os.listdir(dataset_dir,)):
        image_path = os.path.join(dataset_dir,image_name)
        image = cv2.imread(image_path)
        h,w,c = np.shape(image)
        data.append([w,h,h*w])
        if h*w>=2000:
            cv2.imwrite("../action_data/VOC2020_person_area2000+/"+image_name,image)
    data = pd.DataFrame(np.array(data),columns=['w','h','area'],)
    #print(data)
    print(data.describe())
    print(data.head)

def run_main():
    """
    这是主函数
    """
    dataset_dir = os.path.abspath("../action_data/VOC2020_person")
    count_image_resolution(dataset_dir)

    # dataset_dir = os.path.abspath("../../origin_dataset/helmet_detection_dataset1-1")
    # count_image_resolution(dataset_dir)

    # dataset_dir = os.path.abspath("../../origin_dataset/helmet_detection_dataset2-1")
    # count_image_resolution(dataset_dir)

if __name__ == '__main__':
    run_main()
