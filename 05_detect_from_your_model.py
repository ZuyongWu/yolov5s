"""
@File    :   05_detect_from_your_model.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/11 23:28
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Detection import ObjectDetection


def detect_image():
    input_file = "images/facial_mask.jpeg"

    detector = ObjectDetection(weight_path=path_to_weight)
    detector.detectObjectsFromImage(input_file=input_file, out_to_file=None, save_txt=False,
                                    confidence=0.4, custom_class=None)


def detect_video():
    input_file = "videos/facial_mask.mp4"
    out_to_file = "videos/detected/facial_mask.mp4"

    detector = ObjectDetection(weight_path=path_to_weight)
    detector.detectObjectsFromVideo(input_file=input_file, out_to_file=out_to_file, show_progress=True, resized=0.5,
                                    confidence=0.4, custom_class=None)


if __name__ == '__main__':
    path_to_weight = "pre_trained_model/best_facial_mask.pt"
    detect_image()
    # detect_video()
