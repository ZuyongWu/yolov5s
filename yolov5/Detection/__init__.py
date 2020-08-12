"""
@File    :   __init__.py.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/2 20:14
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""

from torch.backends import cudnn
from yolov5.Detection.utils import torch_utils
from yolov5.Detection.utils.datasets import letterbox
from yolov5.Detection.utils.experimental import attempt_load
from yolov5.Detection.utils.utils import check_img_size, non_max_suppression, scale_coords, plot_one_box, xyxy2xywh

import torch
import random
import os
import time
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class ObjectDetection:
    """
    """
    def __init__(self, weight_path, image_size=640, iou_threshold=0.5):
        """
        :param weight_path:
        :param image_size:
        """
        self.weight_path = weight_path
        self._iou_threshold = iou_threshold

        self.device = torch_utils.select_device()
        # Load FP32 model
        self.model = attempt_load(self.weight_path, map_location=self.device)
        # check img_size  640
        self.image_size = check_img_size(image_size, s=self.model.stride.max())
        self.half = self.device.type != 'cpu'

        if self.half:  # half precision only supported on CUDA
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect_per_image(self, input_image, confidence=0.4, custom_class=None):
        """
        :param input_image: BGR format
        :param confidence:
        :param custom_class:
        :return:
        """
        padded_image = letterbox(input_image, new_shape=self.image_size)[0]
        # Convert
        img = padded_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS, pred[0].shape = [1, 6] = [object_nums, 6], len(pred) = batch_size = 1
        pred = non_max_suppression(pred, confidence, self._iou_threshold, classes=custom_class)
        t2 = torch_utils.time_synchronized()

        #                                         -------box coordination ------     conf     class
        # shape = [object_nums, 6], ex: [1, 6] =[[ 26.00, 186.25, 500.00, 489.25,   0.859,   0.000]]
        result = pred[0]

        if result is not None and len(result):
            # Rescale boxes from img_size(512, 640) to real_image size(480, 640)
            result[:, :4] = scale_coords(img.shape[2:], result[:, :4], input_image.shape).round()

            # Write results
            for *xyxy, conf, cls in result:
                label = '{} {:.2f}'.format(self.names[int(cls)], conf)
                plot_one_box(xyxy, input_image, label=label, color=self.colors[int(cls)], line_thickness=2)

        return t2 - t1, result, input_image

    def detectObjectsFromImage(self, input_file=None, out_to_file=None, confidence=0.4,
                               custom_class=None, save_txt=False):
        """
        :param input_file:
        :param out_to_file:
        :param confidence:
        :param custom_class:
        :param save_txt:
        :return:
        """
        assert os.path.isfile(input_file), "input must be a file"
        if out_to_file is None:
            base_dir = "/".join(input_file.split("/")[:-1]) + "/detected"
            os.makedirs(base_dir, exist_ok=True)
            out_to_file = base_dir + "/" + input_file.split("/")[-1]
        txt_path = "".join(out_to_file.split(".")[:-1]) if save_txt else None
        if txt_path and txt_path.startswith("/"):
            txt_path = "".join(txt_path[1:])

        t0 = time.time()

        img0 = cv2.imread(input_file)  # BGR
        run_time, result, img0 = self.detect_per_image(input_image=img0, confidence=confidence, custom_class=custom_class)

        if save_txt:
            gain = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            with open(txt_path + '.txt', 'w') as f:
                for *xyxy, conf, cls in result:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gain).view(-1).tolist()  # normalized xywh
                    f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

        # Save results (image with detections)
        cv2.imwrite(out_to_file, img0)
        print('Done. (%.3fs)' % (time.time() - t0))

    def detectObjectsFromVideo(self, input_file=None, out_to_file=None, show_progress=True,
                               resized=None, confidence=0.4, custom_class=None):
        """
        :param input_file:
        :param out_to_file:
        :param show_progress:
        :param resized:
        :param confidence:
        :param custom_class:
        :return:
        """
        assert os.path.isfile(input_file), "input video must be a file."
        if out_to_file is None:
            out_to_file = "".join(input_file.split(".")[:-1]) + "_detected.mp4"

        # Run inference
        t0 = time.time()

        vid_cap = cv2.VideoCapture(input_file)

        length = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vid_writer = cv2.VideoWriter(out_to_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        count = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if ret:
                count += 1
                real_image = frame.copy()

                run_time, result, real_image = self.detect_per_image(input_image=real_image,
                                                                     confidence=confidence,
                                                                     custom_class=custom_class)

                vid_writer.write(real_image)
                print('\r{}/{}   Process Frame Time: {:.3f}s'.format(count, length, run_time), end='')

                if show_progress:
                    if resized:
                        real_image = cv2.resize(real_image, (0, 0), fx=resized, fy=resized)
                    cv2.imshow("Video", real_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            else:
                cv2.destroyAllWindows()
                break
        print('\nResults saved to {}'.format(out_to_file))
        print('Done. (%.3fs)' % (time.time() - t0))

    def detectObjectsFromCamera(self, out_to_file=None, save_video=False,
                                confidence=0.4, custom_class=None, fps=30):
        """
        :param fps:
        :param out_to_file:
        :param save_video:
        :param confidence:
        :param custom_class:
        :return:
        """
        if save_video:
            assert out_to_file, "out_to_file must be not None."

        vid_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if save_video:
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(out_to_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        else:
            vid_writer = None

        count = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if ret:
                count += 1
                real_image = frame.copy()
                run_time, result, real_image = self.detect_per_image(input_image=real_image,
                                                                     confidence=confidence,
                                                                     custom_class=custom_class)
                print('\rFrame: {},  Process Frame Time: {:.3f}s'.format(count, run_time), end='')

                if save_video:
                    vid_writer.write(real_image)

                cv2.imshow("Video", real_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if save_video:
            print('\nResults saved to {}'.format(out_to_file))
