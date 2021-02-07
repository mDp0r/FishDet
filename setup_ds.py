import os
from PIL import Image
import numpy as np
from kmeans_anchors_ratios import get_optimal_anchors_ratios
import json
from shutil import copytree, copy2, rmtree
import urllib.request
import tarfile

"""
Generell: 



Args einlesen
__________________________________________
V1: Datensatz liegt in COCO Format vor ->mode "coco"

Projekt yml erzeugen und unter projects schreiben
    -aus Args
    -AVG RGB berechnen
    -Anchor ratios mit kmeans
    -obj_list aus Annos erzeugen
Directories erstellen (unter Yet..../datasets/
Ordner an richtige Stelle kopieren

__________________________________________
V2: Datensatz liegt nicht in COCO Format vor und ist keiner der vorgegebenen Datensätze -> mode "any"
-if not exist: data_to_infer Ordner erstellen
-in Ordner data_to_infer kopieren
__________________________________________
V3: Datensatzname ist deepfish
-Annos und Projekt File aus util files laden und in Yet Another... Schieben -> mode "known"
-download 
-unzip
-do train test split from Annos

__________________________________________
V4: Format in LOST -> mode "lost"
-Annos einlesen
-Annos zu COCO konvertieren
-mit COCO Creator laufen lassen mit temp_path
-Cleanup
"""

class coco_creator:
    
    def __init__(self, args):
        self.c = args.c
        self.ds_name = args.ds_name
        self.path = args.path
    
        with open(self.path + "annotations/instances_train.json") as f:
            self.instances = json.load(f)
        return
    
    def get_avg_rgb_single(self, impath):
        arr = np.asarray(Image.open(impath))
        return arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2]).mean(axis=0)
    
    def get_rgb_values(self):
        print("Calculating RGB mean and std for train set")
        arr_list = []
        for img in os.listdir(self.path+"train/"):
            temp_arr = self.get_avg_rgb_single(self.path+"train/"+img)
            arr_list.append(temp_arr)
        arr = np.asarray(arr_list)
        arr /= 255
        self.rgb_mean = arr.mean(axis=0).tolist()
        self.rgb_std = arr.std(axis=0).tolist()
        return
    
    def get_efficientdet_anchors(self):
        print("Calculating KMeans Anchors")
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0]

        scale = anchor_scale[self.c]
        strides = 2 ** np.arange(3, pyramid_levels[self.c] + 3)
        self.scales_string = "[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]"
        scales = np.array(eval(self.scales_string))

        INPUT_SIZE = input_sizes[self.c]
        ANCHORS_SIZES = (scale * scales * strides[:, np.newaxis]).flatten().tolist()


        ratios = get_optimal_anchors_ratios(
            self.instances,
            anchors_sizes=ANCHORS_SIZES,
            input_size=INPUT_SIZE,
            normalizes_bboxes=True,
            num_runs=100,
            num_anchors_ratios=3,
            max_iter=2500,
            iou_threshold=0.5,
            min_size=0,
            decimals=1,
            default_anchors_ratios=[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]
        )
        self.scales = scales
        self.ratios_string = str(ratios)
        return 
    
    def get_obj_list(self):
        print("Extracting obj list from annotations")
        self.obj_list = [x["name"] for x in self.instances["categories"]]
        return
    
    def get_output_dic(self):
        print("Creating Output dic")
        self.output_dic = {"project_name":self.ds_name, "train_set":"train", "val_set":"val",\
                          "mean":self.rgb_mean, "std":self.rgb_std, \
                          "anchors_ratios": self.ratios_string, "anchors_scales" : self.scales_string, 
                           "num_gpus":1, "obj_list":self.obj_list\
                          }
        return
    
    
    def write_dic_to_yaml(self):
        print("Writing File")
        if "projects" not in os.listdir("Yet-Another-EfficientDet-Pytorch/"):
            os.mkdir("Yet-Another-EfficientDet-Pytorch/projects")
            
        with open(f'Yet-Another-EfficientDet-Pytorch/projects/{self.ds_name}.yml', 'w') as file:
            yaml.dump(self.output_dic, file)
        
    
    def copy_images(self):
        print("Copying images to efficientdet folder")
        if "datasets" not in os.listdir("Yet-Another-EfficientDet-Pytorch/"):
            os.mkdir("Yet-Another-EfficientDet-Pytorch/datasets")
        copytree(self.path, f"Yet-Another-EfficientDet-Pytorch/datasets/{self.ds_name}")
        
        
    
    def do_pipeline(self):
        self.get_efficientdet_anchors()
        self.get_rgb_values()
        self.get_obj_list()
        self.get_output_dic()
        self.write_dic_to_yaml()
        self.copy_images()
            
class any_creator:
    
    def __init__(self, args):
        self.ds_name = args.ds_name
        self.path = args.path
        return
    
    def copy_images(self):
        print("Copying images to efficientdet folder")
        if "data_to_infer" not in os.listdir("Yet-Another-EfficientDet-Pytorch/"):
            os.mkdir("Yet-Another-EfficientDet-Pytorch/data_to_infer")
        copytree(self.path, f"Yet-Another-EfficientDet-Pytorch/data_to_infer/{self.ds_name}")
    
    def do_pipeline(self):
        self.copy_images()
        
        
    
    
    
class known_creator:
    
    def __init__(self,args):
        self.ds_name = args.ds_name
        return
    
    def do_pipeline(self):
        if self.ds_name == "deepfish":
            self.create_deepfish()
        else:
            print("No known dataset specified. Aborting.")
        return
    
    def create_deepfish(self):
        self.anno_train_pth = "util_files/deepfish_annos/annotations/instances_train.json"
        self.anno_val_pth = "util_files/deepfish_annos/annotations/instances_val.json"   
        self.download_deepfish()
        self.unzip_deepfish()
        self.copy_images()
        self.copy_util_files() # annos and project yml
        self.cleanup()
        return
    
    def download_deepfish(self):
        print('Downloading Deepfish Dataset')
        url = 'https://cloudstor.aarnet.edu.au/plus/s/NfjObIhtUYO6332/download'
        urllib.request.urlretrieve(url, filename = "deepfish.tar")
        
    def unzip_deepfish(self):
        tar = tarfile.open("deepfish.tar", "r:")
        tar.extractall()
        tar.close()
        
    def copy_images(self):
        print("Copying images to efficientdet folder")
        if "datasets" not in os.listdir("Yet-Another-EfficientDet-Pytorch/"):
            os.mkdir("Yet-Another-EfficientDet-Pytorch/datasets")
        os.mkdir("Yet-Another-EfficientDet-Pytorch/datasets/deepfish")
        os.mkdir("Yet-Another-EfficientDet-Pytorch/datasets/deepfish/train")
        os.mkdir("Yet-Another-EfficientDet-Pytorch/datasets/deepfish/val")
        with open(self.anno_train_pth) as f:
            train_instances = json.load(f)
            
        train_img_list = [x["file_name"] for x in train_instances["images"]]
        
        with open(self.anno_val_pth) as f:
            val_instances = json.load(f)
            
        val_img_list = [x["file_name"] for x in val_instances["images"]]
        
        for img in os.listdir("DeepFish/Segmentation/images/valid/"):
            if img in train_img_list:
                copy2("DeepFish/Segmentation/images/valid/"+img, "Yet-Another-EfficientDet-Pytorch/datasets/deepfish/train/")
            elif img in val_img_list:
                copy2("DeepFish/Segmentation/images/valid/"+img, "Yet-Another-EfficientDet-Pytorch/datasets/deepfish/val/")
        return
    
    def copy_util_files(self):
        copy2("util_files/deepfish.yml", "Yet-Another-EfficientDet-Pytorch/projects/")
        copytree("util_files/deepfish_annos/annotations/", "Yet-Another-EfficientDet-Pytorch/datasets/deepfish/annotations/")
        return
    
    def cleanup(self):
        os.remove("deepfish.tar")
        rmtree("DeepFish" )
        return


def get_inputs():
    #Initialize parser
    parser = argparse.ArgumentParser(description='Script to setup Dataset in EfficientDet Repo')
    parser.add_argument("--c", action="store", dest = "c", type=int, default = None)
    parser.add_argument("--ds_name", action="store", dest = "ds_name", default = None)
    parser.add_argument("--path", action="store", dest="path", type=str, \
default = None)
    parser.add_argument("--mode", action="store", dest="mode", type=str, \
default = "coco")
    #Return parsed object
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = get_inputs()
    
    if parsed_args.mode == "coco":
        creator = coco_creator(parsed_args)
        
    elif parsed_args.mode == "any":
        creator = any_creator(parsed_args)
        
    elif parsed_args.mode =="known":
        creator = known_creator(parsed_args)
        
    else:
        print("No proper mode specified. Aborting")
    creator.do_pipeline()
    
    