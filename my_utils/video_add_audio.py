"""
@File    :   video_add_audio.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/3/23 15:10
@Author  :   Bruce Wu
@Version :   1.0 
@Des     :   pip install moviepy
"""

from moviepy.editor import VideoFileClip, AudioFileClip
import os


def add_audio_to_video(video_path_audio, video_path_no_audio):
    """
    Add audio of video_path_audio to video_path_no_audio, creating a new video file contain audio

    Args:
        video_path_audio (string):      video file contain audio, like "video/outside_wander.mp4"
        video_path_no_audio (string):   video file no contain audio, like "video/outside_wander_detection.avi"

    Returns: None
    """
    origin_path = video_path_no_audio
    if video_path_no_audio.split(".")[-1] == "avi":
        new_name = video_path_no_audio.split(".")[0] + ".mp4"
        os.rename(video_path_no_audio, new_name)
        video_path_no_audio = new_name

    audio_path = video_path_audio.split(".")[0] + ".mp3"
    new_video_path = video_path_no_audio.split(".")[0] + "_new." + video_path_no_audio.split(".")[1]

    video = VideoFileClip(video_path_audio)
    video.audio.write_audiofile(audio_path)

    video_no_audio = VideoFileClip(video_path_no_audio)
    audio = AudioFileClip(audio_path)

    new_video = video_no_audio.set_audio(audio)
    new_video.write_videofile(new_video_path)

    os.remove(audio_path)
    if origin_path != video_path_no_audio:
        os.rename(video_path_no_audio, origin_path)
    print(new_video_path + " has been saved.")


if __name__ == '__main__':
    path_to_file_audio = ".mp4"
    path_to_file_no_audio = ".mp4"
    add_audio_to_video(path_to_file_audio, path_to_file_no_audio)
