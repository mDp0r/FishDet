import os
from PIL import Image
import numpy as np
from kmeans_anchors_ratios import get_optimal_anchors_ratios
import json
from shutil import copytree, copy2, rmtree
import urllib.request
import tarfile
from tqdm import tqdm
from copy import copy as copy_obj
import yaml
from ast import literal_eval
import argparse
import pandas as pd
import datetime
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
            normalizes_bboxes=False,
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
        print("Unzipping")
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
        print("Copying util files")
        copy2("util_files/deepfish.yml", "Yet-Another-EfficientDet-Pytorch/projects/")
        copytree("util_files/deepfish_annos/annotations/", "Yet-Another-EfficientDet-Pytorch/datasets/deepfish/annotations/")
        return
    
    def cleanup(self):
        print("Cleaning up")
        os.remove("deepfish.tar")
        rmtree("DeepFish" )
        return

    
class lost_creator:
    def __init__(self, args):
        self.args = copy_obj(args)
        self.anno_file = args.anno_file
        self.annos = pd.read_csv(self.args.path+ "annos_out/"+self.anno_file)
        self.img_path = self.args.path + "imgs/"
        return
    
    def __create_infos(self):
        #Infoteil
        self.info_dic={"year":datetime.datetime.today().year,\
        "version":"",\
        "description":f"Annotations generated from LOST. Project_Name {self.args.ds_name}",\
        "contributor":"", "url":"", "date_created":datetime.datetime.today()}
        self.info_json=json.dumps(self.info_dic, default=str) 
        return
    
    def __create_licenses(self):
        #Infoteil
        self.licenses_list=[{"id":1, "name":"Free Licence", "url":""}]
        self.licenses_json=json.dumps(self.licenses_list, default=str) 
        return
    
    def __create_categories(self):
        #Infoteil
        self.cat_list = []
        for label in self.annos.groupby(["anno.lbl.name", "anno.lbl.external_id"]).size().index:
            self.cat_list.append({"id":label[1]\
                             , "name":label[0], "supercategory":""})
        self.cat_json=json.dumps(self.cat_list)
        return
    
    def __train_val_split(self):
        imgs = self.annos["img.img_path"].unique()
        train_imgs = imgs[:int(np.around(len(imgs)/2))]
        
        self.train = self.annos.loc\
        [self.annos["img.img_path"].isin(train_imgs)].copy().reset_index(drop = True)
        
        self.val = self.annos.loc\
        [~self.annos["img.img_path"].isin(train_imgs)].copy().reset_index(drop = True)
        

        return

    def __generate_img_metadata(self, df):
        df = df.loc[:, ["file_name"]].copy()
        df = df.drop_duplicates()
        df["width"] = df["file_name"].apply(lambda x: Image.open(self.img_path + x).width)
        df["height"] = df["file_name"].apply(lambda x: Image.open(self.img_path + x).height)
        df["license"] = 1
        df = df.reset_index(drop=True).reset_index().rename(columns={"index":"id"})
        return df
    
    def __process_annos(self,df):
        df = df.loc[:,["file_name", "width", "height", "bbox", "anno.lbl.external_id"]].copy()
        df["area"] = df["width"].apply(lambda x: round(x)) * df["height"].apply(lambda x: round(x))
        df = df.drop(columns=["width","height"])
        df = df.rename(columns = {"anno.lbl.external_id": "category_id"})
        df["iscrowd"] = 0
        df["segmentation"]=np.empty((len(df), 0)).tolist()
        df = df.reset_index(drop=True).reset_index().rename(columns={"index":"id"})
        return df
    
    def __format_bbox_coco(self, annos):
        annos["x"] = annos["anno.data"].apply(lambda x: literal_eval(x)["x"])
        annos["y"] = annos["anno.data"].apply(lambda x: literal_eval(x)["y"])
        annos["width"] = annos["anno.data"].apply(lambda x: literal_eval(x)["w"])
        annos["height"] = annos["anno.data"].apply(lambda x: literal_eval(x)["h"])
        annos["x"] = annos["x"].copy()-0.5*annos["width"].copy()
        annos["y"] = annos["y"].copy()-0.5*annos["height"].copy()
        annos["im_width"] = annos["file_name"].apply(lambda x: Image.open(self.img_path + x).width)
        annos["im_height"] = annos["file_name"].apply(lambda x: Image.open(self.img_path + x).height)
        annos["x"] = annos["x"].copy() * annos["im_width"].copy()
        annos["width"] = annos["width"].copy() * annos["im_width"].copy()
        annos["y"] = annos["y"].copy() * annos["im_height"].copy()
        annos["height"] = annos["height"].copy() * annos["im_height"].copy()

        annos["x"] = annos["x"].round().astype(int)
        annos["y"] = annos["y"].round().astype(int)
        annos["width"] = annos["width"].round().astype(int)
        annos["height"] = annos["height"].round().astype(int)
        annos = annos.loc[((annos["x"]+annos["width"])< (annos["im_width"])) & ((annos["x"]-annos["width"])>0)].copy()
        annos = annos.loc[((annos["y"]+annos["height"])< (annos["im_height"])) & ((annos["y"]-annos["height"])>0)].copy()
        annos["bbox"] = annos.apply(lambda x: [x["x"], x["y"], x["width"],x["height"]] , axis=1)
        return annos
    
    #labels
    #Annos speichern
    #Bilder kopieren
    def __get_annos_and_img_metadata(self, data):
        imgs_df = self.__generate_img_metadata(data)
        anno_df = self.__process_annos(data)
        imgs_df, anno_df = self.__replace_name_with_id(imgs_df, anno_df)
        return imgs_df, anno_df, imgs_df.to_json(orient="records"),\
        anno_df.to_json(orient="records")
    
    def __replace_name_with_id(self, imdf, annodf):
        annodf = annodf\
        .merge(imdf.rename(columns={"id":"image_id"})\
        .loc[:,["file_name","image_id"]], how="left", on=["file_name"])\
        .drop(columns=["file_name"])
        return imdf, annodf
    
    def __create_temp_dir(self):
        os.mkdir(self.args.ds_name)
        os.mkdir(self.args.ds_name+ "/annotations")
        os.mkdir(self.args.ds_name+ "/train")
        os.mkdir(self.args.ds_name+ "/val")
        return
    def __create_full_json(self,info, licenses, cat, images, annos, filename):
        full_json = {"info": json.loads(info), \
        "licenses" :   json.loads(licenses),\
        "categories":  json.loads(cat)  , \
        "images":  json.loads(images)  , \
        "annotations" :  json.loads(annos)  , \
                    }
        with open(self.args.ds_name+"/annotations/"+filename, "w")\
        as outfile:
            json.dump(full_json, outfile)
        return
    def __cleanup(self):
        rmtree(self.args.ds_name)
        return
    
    def __copy_images(self):
            for idx, file_name in tqdm(self.train_img_df["file_name"].iteritems()):
                    copy2(self.img_path+file_name, self.args.ds_name + "/train/")
            for idx, file_name in tqdm(self.val_img_df["file_name"].iteritems()):
                    copy2(self.img_path+file_name, self.args.ds_name + "/val/")
    
    def do_pipeline(self):
        self.__create_temp_dir()
        self.__create_infos()
        self.__create_licenses()
        self.annos["anno.lbl.name"] = self.annos["anno.lbl.name"].apply(lambda x: literal_eval(x)).str[0]
        self.annos["anno.lbl.external_id"] = self.annos["anno.lbl.external_id"].apply(lambda x: literal_eval(x)).str[0]
        self.annos = self.annos.loc[self.annos["anno.lbl.external_id"].notna()].copy()
        self.annos["anno.lbl.external_id"] = self.annos["anno.lbl.external_id"].astype(int)
        self.annos["file_name"] = self.annos["img.img_path"].str.split("/").str[-1]
        self.__create_categories()
        self.annos = self.__format_bbox_coco(self.annos)
        self.__train_val_split()
        
        self.train_img_df, self.train_anno_df, self.train_img_json, self.train_anno_json = \
        self.__get_annos_and_img_metadata(self.train)
        
        self.val_img_df, self.val_anno_df, self.val_img_json, self.val_anno_json = \
        self.__get_annos_and_img_metadata(self.val)
        
        self.__create_full_json(self.info_json, self.licenses_json, self.cat_json,\
        self.train_img_json, self.train_anno_json, "instances_train.json")
        
        self.__create_full_json(self.info_json, self.licenses_json, self.cat_json,\
        self.val_img_json, self.val_anno_json, "instances_val.json")
        
        self.args.path = self.args.ds_name + "/"
        self.__copy_images()
        temp_coco_creator = coco_creator(self.args)
        temp_coco_creator.do_pipeline()
        self.__cleanup()
        
        return
    


def get_inputs():
    #Initialize parser
    parser = argparse.ArgumentParser(description='Script to setup Dataset in EfficientDet Repo')
    parser.add_argument("--c", action="store", dest = "c", type=int, default = None)
    parser.add_argument("--ds_name", action="store", dest = "ds_name", default = None)
    parser.add_argument("--path", action="store", dest="path", type=str, \
default = None)
    parser.add_argument("--mode", action="store", dest="mode", type=str, \
default = None)
    #Only needed for LOST annos
    parser.add_argument("--anno_file", action = "store", dest="anno_file", type=str, \
    default = None)
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
    
    elif parsed_args.mode == "lost":
        creator = lost_creator(parsed_args)
        
    else:
        print("No proper mode specified. Aborting")
    creator.do_pipeline()
    

# class test_class:
#     def __init__(self, *args, **kwargs):
#         self.__dict__.update(kwargs)
        
# test_dic = {"mode":"coco", "path":"/home/mario/Test_FishDet/FishDet/fishes_2103231818/","c":4, "ds_name":"fishes_2103231818"}
# parsed_args = test_class(**test_dic)
# if parsed_args.mode == "coco":
#     creator = coco_creator(parsed_args)
    
# elif parsed_args.mode == "any":
#     creator = any_creator(parsed_args)
    
# elif parsed_args.mode =="known":
#     creator = known_creator(parsed_args)

# elif parsed_args.mode == "lost":
#     creator = lost_creator(parsed_args)
    
# else:
#     print("No proper mode specified. Aborting")
# creator.do_pipeline()
    
    
    