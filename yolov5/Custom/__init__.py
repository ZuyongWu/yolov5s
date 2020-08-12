"""
@File    :   __init__.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/2 20:14
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Custom.utils.makeTxt import make_txt
from yolov5.Custom.utils.makeLabel import make_label
from yolov5.Custom.utils.makeConfigFile import make_config_file
from yolov5.Custom.train import train
import os
import argparse


class Custom_Object_Detect_Training:
    """"""
    def __init__(self):
        self.object_names = None
        self.data_directory = None
        self.pretrain_model = None
        self.batch_size = None
        self.epochs = None

    def setDataDirectory(self, data_directory=None, object_names=None):
        self.object_names = object_names
        self.data_directory = data_directory

        make_txt(data_directory=data_directory)
        make_label(data_directory=data_directory, object_names=object_names)
        make_config_file(data_directory=data_directory, object_names=object_names)

    def setTrainConfig(self, pretrain_model=None, batch_size=8, epochs=300):
        self.pretrain_model = pretrain_model
        self.batch_size = batch_size
        self.epochs = epochs

    def trainModel(self):
        opt = argparse.ArgumentParser().parse_args()
        opt.cfg = self.data_directory + '/config_file/model.yaml'
        opt.data = self.data_directory + '/config_file/data.yaml'
        opt.weights = self.pretrain_model
        opt.name = self.data_directory
        opt.epochs = self.epochs
        opt.batch_size = self.batch_size

        train(opt)
