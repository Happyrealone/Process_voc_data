import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

def get_data_from_xml(file="2011_003285.xml"):
    # 从图片中获取动作与框位置
    file = open(file)
    tree = ET.parse(file)
    root = tree.getroot()

    filename = root.find("filename").text
    acts = []
    axes = []
    for object in root.iter('object'):
        if object.find("actions"):
            actions = object.find("actions")
            for action in actions:
                if action.tag in ["walking", "running"]:
                    if action.text == "1":
                        acts.append(action.tag)
                        break
            boxs = object.find("bndbox")
            xmax = int(float(boxs.find("xmax").text))
            xmin = int(float(boxs.find("xmin").text))
            ymax = int(float(boxs.find("ymax").text))
            ymin = int(float(boxs.find("ymin").text))

            axes.append([ymin, ymax, xmin, xmax])

    if len(acts) > 0:
        return filename, acts, axes

# print(get_data_from_xml())

def cut_from_img(xml_path = None):
    # 使用xml文件中获取到的信息切割图片，放置在当前目录下
    if get_data_from_xml(file=xml_path):
        filename, acts, axes = get_data_from_xml(file=xml_path)
        for i in range(len(acts)):
            action = acts[i]
            axe = axes[i]
            img = cv2.imread(imgs_path + filename)
            result_file_name = filename.split(".")[0] + "_" + "{}.jpg".format(i)
            cv2.imwrite(action + "/" + result_file_name, img[axe[0]:axe[1], axe[2]:axe[3]])
        
if __name__ == '__main__':
    
    imgs_path = "../action_data/VOCdevkit/VOC2012/JPEGImages/"
    xmls_path = "../action_data/VOCdevkit/VOC2012/Annotations/"
    for act in ["./running","./walking"]:
        if not os.path.exists(act) : 
            os.makedirs(act)

    for xml_name in tqdm(os.listdir(xmls_path)):
        xml_path = xmls_path + xml_name
        cut_from_img(xml_path=xml_path)