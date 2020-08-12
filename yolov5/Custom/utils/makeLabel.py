"""
@File    :   makeLabel.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/7 22:54
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import xml.etree.ElementTree as ET
import pickle
import os


def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1./size[0]     # 1/w
    dh = 1./size[1]     # 1/h
    x = (box[0] + box[1])/2.0   # 物体在图中的中心点x坐标
    y = (box[2] + box[3])/2.0   # 物体在图中的中心点y坐标
    w = box[1] - box[0]         # 物体实际像素宽度
    h = box[3] - box[2]         # 物体实际像素高度
    x = x*dw    # 物体中心点x的坐标比(相当于 x/原图w)
    w = w*dw    # 物体宽度的宽度比(相当于 w/原图w)
    y = y*dh    # 物体中心点y的坐标比(相当于 y/原图h)
    h = h*dh    # 物体宽度的宽度比(相当于 h/原图h)
    return x, y, w, h  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


def convert_annotation(input_file_path, output_file_path, class_names):
    """
    """

    # 对应的通过year 找到相应的文件夹，并且打开相应image_id的xml文件，其对应bounding文件
    in_file = open(input_file_path, encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open(output_file_path, 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对
    root = tree.getroot()
    # 获得图片的尺寸大小
    size = root.find('size')
    # 获得宽
    w = int(size.find('width').text)
    # 获得高
    h = int(size.find('height').text)
    # 遍历目标obj
    for obj in root.iter('object'):
        # 获得类别 =string 类型
        cls = obj.find('name').text
        # 如果类别不是对应在我们预定好的class文件中跳过
        if cls not in class_names:
            continue
        # 通过类别名称找到id
        cls_id = class_names.index(cls)
        # 找到bndbox 对象
        xmlbox = obj.find('bndbox')
        # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        # 带入进行归一化操作
        # w = 宽, h = 高， b= bound-box的数组 = ['xmin','xmax','ymin','ymax']
        bb = convert((w, h), b)
        # bb 对应的是归一化后的(x,y,w,h)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def make_label(data_directory, object_names):
    """
    :param object_names:
    :param data_directory:
    :return:
    """
    label_dir = os.path.join(data_directory, "labels")
    os.makedirs(label_dir, exist_ok=True)

    annotation_dir = os.path.join(data_directory, "annotations")
    annotation_file_list = os.listdir(annotation_dir)

    for annotation_file in annotation_file_list:
        annotation_file_path = os.path.join(annotation_dir, annotation_file)
        annotation_file_name = annotation_file.split(".")[0]

        label_file_name = str(annotation_file_name) + ".txt"
        label_file_path = os.path.join(label_dir, label_file_name)

        convert_annotation(annotation_file_path, label_file_path, object_names)
