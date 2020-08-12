"""
@File    :   makeConfigFile.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/8 15:29
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import yaml
import os
import shutil
import re


def make_config_file(data_directory, object_names):
    config_dir = os.path.join(data_directory, "config_file")
    os.makedirs(config_dir, exist_ok=True)

    data_yaml_path = os.path.join(config_dir, "data.yaml")
    model_yaml_path = os.path.join(config_dir, "model.yaml")

    if not os.path.exists(data_yaml_path):
        data_yaml = {
            "train": data_directory + "/txt_file/train.txt",
            "val": data_directory + "/txt_file/val.txt",
            "nc": len(object_names),
            "names": object_names
        }

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

    if not os.path.exists(model_yaml_path):
        shutil.copy("models/yolov5s.yaml", model_yaml_path)

        with open(model_yaml_path, encoding="utf-8") as f:
            model_yaml = f.read()

        model_yaml = re.sub(r'[0-9]+', str(len(object_names)), model_yaml, count=1)

        with open(model_yaml_path, "w", encoding="utf-8") as f:
            f.write(model_yaml)
