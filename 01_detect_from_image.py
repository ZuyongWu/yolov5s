"""
@File    :   01_detect_from_image.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/11 22:22
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Detection import ObjectDetection


path_to_weight = "pre_trained_model/yolov5s.pt"
input_file = "images/bus.jpg"

detector = ObjectDetection(weight_path=path_to_weight)
detector.detectObjectsFromImage(input_file=input_file, out_to_file=None, save_txt=False,
                                confidence=0.4, custom_class=[5])  # custom_class=None or [0, 1, 2 ...]
