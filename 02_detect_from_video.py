"""
@File    :   02_detect_from_video.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/11 22:31
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov5.Detection import ObjectDetection
from my_utils.video_add_audio import add_audio_to_video


path_to_weight = "pre_trained_model/yolov5s.pt"

input_file = "videos/roads.mp4"
out_to_file = "videos/detected/roads_yolov5s.mp4"

detector = ObjectDetection(weight_path=path_to_weight)
detector.detectObjectsFromVideo(input_file=input_file, out_to_file=out_to_file, show_progress=True, resized=0.5,
                                confidence=0.4, custom_class=None)

# add_audio_to_video(video_path_audio=input_file, video_path_no_audio=out_to_file)
