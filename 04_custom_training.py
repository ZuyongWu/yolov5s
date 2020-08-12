"""
@File    :   04_custom_training.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/7 10:43
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Custom import Custom_Object_Detect_Training


if __name__ == '__main__':
    path_to_model = "pre_trained_model/yolov5s.pt"

    trainer = Custom_Object_Detect_Training()

    trainer.setDataDirectory(data_directory="facial_mask", object_names=["facial mask"])
    trainer.setTrainConfig(batch_size=8, epochs=100, pretrain_model=path_to_model)

    trainer.trainModel()
