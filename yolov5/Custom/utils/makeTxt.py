"""
@File    :   makeTxt.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/7 22:52
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import os
import random


def make_txt(data_directory, data_split=0.1):
    """

    :param data_split:
    :param data_directory:
    :return:
    """
    images_path = os.path.join(data_directory, "images")
    images_file_list = os.listdir(images_path)
    num_of_images = len(images_file_list)

    txt_file_path = os.path.join(data_directory, "txt_file")
    os.makedirs(txt_file_path, exist_ok=True)

    val_num = int(num_of_images * data_split)
    val_ints = random.sample(range(num_of_images), val_num)

    train_file = open(os.path.join(txt_file_path, 'train.txt'), 'w')
    val_file = open(os.path.join(txt_file_path, 'val.txt'), 'w')

    for images_file in images_file_list:
        index_images = images_file_list.index(images_file)
        path_to_image = os.path.join(images_path, images_file) + '\n'
        if index_images in val_ints:
            val_file.write(path_to_image)
        else:
            train_file.write(path_to_image)

    train_file.close()
    val_file.close()
