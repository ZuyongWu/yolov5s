"""
@File    :   03_detect_from_camera.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/11 23:20
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Detection import ObjectDetection


path_to_weight = "pre_trained_model/yolov5s.pt"

detector = ObjectDetection(weight_path=path_to_weight)
detector.detectObjectsFromCamera(out_to_file="videos/detected/camera_detection.mp4", save_video=True,
                                 confidence=0.4, custom_class=None)
