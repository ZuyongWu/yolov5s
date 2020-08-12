
## YOLO V5s Detecting and Training

Pretrained model weights can download by : https://pan.baidu.com/s/1UqODFKeKrn0jqRSO8iPTKg, verified code is:  **yolo**.  Downloaded weights should be placed in **"pre_trained_model"** directory in root of yolov5s directory. Also, downloaded videos should be placed in **"videos"** directory in root of yolov5s directory.




## Detecting

```bash
$ python 01_detect_from_image.py
```

Results are saved to `./images/detected/`.



## Training

FIrstly, you should preparing your custom dataset. Our model need **Pascal VOC format** for image annotation. You can generate this annotation for your images using the easy to use [**LabelImg**](https://github.com/tzutalin/labelImg) image annotation tool. See: https://github.com/tzutalin/labelImg.

The structure of your image dataset folder should look like below (placing in root of **"yolov5s"** directory):

```python
.  >> dataset_directory    >> images       >> img_1.jpg
                                           >> img_2.jpg
                                           >> img_3.jpg
 
                           >> annotations  >> img_1.xml
                                           >> img_2.xml
                                           >> img_3.xml
```
After that, editing 04_custom_training.py with your custom configuration:

```python
path_to_model = "pre_trained_model/yolov5s.pt"
trainer = Custom_Object_Detect_Training()
trainer.setDataDirectory(data_directory="dataset diretory", object_names=["names"])
trainer.setTrainConfig(batch_size=8, epochs=100, pretrain_model=path_to_model)

trainer.trainModel()
```

Results will be saved in **"runs"** diretory. Then you can use trained weights for your custom detecting.




## My Environment

Windows System installed with CUDA 10.1 and Python 3.6.8, other dependencies are needed below:

- ```python
  torch==1.5.1+cu101
  torchvision==0.6.1+cu101
  
  Note: pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f
        https://download.pytorch.org/whl/torch_stable.html
        or select other version in Pytorch website.
  ```

- ```python
  numpy==1.19.1、opencv-python、 matplotlib
  ```

- ```python
  pillow、 tensorboard、 scipy、 tqdm、 moviepy
  ```

- ```python
  apex
  -- download in https://github.com/NVIDIA/apex
  -- install by: unzip package
                 "cd apex"
                 "pip install -v --no-cache-dir ."
  ```



## Contact

**Issues should be raised directly in the repository.** For more details can email me at 156618056@qq.com. 



## References

1、https://github.com/ultralytics/yolov5

2、https://github.com/williamhyin/yolov5s_bdd100k
