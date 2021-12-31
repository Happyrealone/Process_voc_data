import multiprocessing
from multiprocessing import cpu_count
import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm
import numpy

def fix_axe(image_shape,axe):
    if axe[0]<0:
        axe[0]=0
    if axe[1]>image_shape[0]:
        axe[1]=image_shape[0]
    if axe[2]<0:
        axe[2]=0
    if axe[3]>image_shape[1]:
        axe[3]=image_shape[1]
    return axe


def get_data_from_xml(file=None):
    # 从图片中获取动作与框位置
    file = open(file)
    tree = ET.parse(file)
    root = tree.getroot()

    filename = root.find("filename").text

    axes = []
    for object in root.iter('object'):
        if object.find("name").text == "fall":
            boxs = object.find("bndbox")
            xmax = int(float(boxs.find("xmax").text))
            xmin = int(float(boxs.find("xmin").text))
            ymax = int(float(boxs.find("ymax").text))
            ymin = int(float(boxs.find("ymin").text))
            axes.append([ymin, ymax, xmin, xmax])

    return filename, axes


def cut_from_img(xml_path=None, target_path=None, imgs_path=None):
    # 使用xml文件中获取到的信息切割图片，放置在当前目录下
    filename, axes = get_data_from_xml(file=xml_path)
    for i in range(len(axes)):
        axe = axes[i]
        img = cv2.imread(imgs_path + filename)
        result_file_name = filename.split(".")[0] + "_" + "{}.jpg".format(i)
        axe = fix_axe(img.shape,axe)
        cv2.imwrite(target_path + result_file_name, img[axe[0]:axe[1], axe[2]:axe[3]])


def thread_process(xmls_path, target_path=None, imgs_path=None):
    for xml_name in tqdm(xmls_path):
        cut_from_img(xml_name, target_path, imgs_path)


if __name__ == '__main__':

    imgs_paths = ["../action_data/Fall/JPEGImages/"]
    xmls_paths = ["../action_data/Fall/Annotations/"]
    target = "../action_data/"

    num_thread = cpu_count() - 2

    # 多线程
    for imgs_path, xmls_path in zip(imgs_paths, xmls_paths):
        target_path = target + imgs_path.split("/")[-3] + "_person/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # 按照进程数划分数据集
        xml_path_list = []
        for xml_name in os.listdir(xmls_path):
            xml_path_list.append(xmls_path + xml_name)

        # 每个进程处理的步数
        step = int(len(xml_path_list) / num_thread)
        pool = multiprocessing.Pool(processes=num_thread)

        for i in range(num_thread):
            pool.apply_async(func=thread_process, args=(xml_path_list[i * step:(i + 1) * step], target_path, imgs_path))
        pool.close()
        pool.join()

    # for file in os.listdir("../action_data/Fall/Annotations/"):
    #     cut_from_img(xml_path="../action_data/Fall/Annotations/" + file,
    #                  target_path="../action_data/Fall_person",
    #                  imgs_path="../action_data/Fall/JPEGImages/")
..