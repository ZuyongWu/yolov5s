"""
@File    :   06_detect_from_transfer_model.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/11 23:30
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Detection import ObjectDetection


def detect_image():
    input_file = "images/road.jpg"

    detector = ObjectDetection(weight_path=path_to_weight)
    detector.detectObjectsFromImage(input_file=input_file, out_to_file=None, save_txt=False,
                                    confidence=0.4, custom_class=None)


def detect_video():
    input_file = "videos/roads.mp4"
    out_to_file = "videos/detected/roads_bdd.mp4"

    detector = ObjectDetection(weight_path=path_to_weight)
    detector.detectObjectsFromVideo(input_file=input_file, out_to_file=out_to_file, show_progress=True, resized=0.5,
                                    confidence=0.4, custom_class=[7, 8, 9, 10, 11])


if __name__ == '__main__':
    path_to_weight = "pre_trained_model/best_bdd.pt"
    detect_image()
    # detect_video()
