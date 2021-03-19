import argparse
from setup_ds import coco_creator
import pandas as pd
import json
import os
import datetime
from shutil import copy2, rmtree
from tqdm import tqdm
from copy import copy as copy_obj

class combiner:
    def __init__(self, args):
        self.ds1 = args.ds1
        self.ds2 = args.ds2
        self.ds_name = args.ds_name
        self.args = copy_obj(args)
        del self.args.ds1
        del self.args.ds2
        self.args.path = self.ds_name +"/"
        

        self.path1 = "Yet-Another-EfficientDet-Pytorch/datasets/" + self.ds1 +"/"
        self.path2 = "Yet-Another-EfficientDet-Pytorch/datasets/" + self.ds2 +"/"
        
        
        self.c = args.c

        self.train_annos_json1 = self.read_anno_json(self.path1+"annotations/instances_train.json")
        self.train_annos_json2 = self.read_anno_json(self.path2+"annotations/instances_train.json")
        
        self.val_annos_json1 = self.read_anno_json(self.path1+"annotations/instances_val.json")
        self.val_annos_json2 = self.read_anno_json(self.path2+"annotations/instances_val.json")
        
        self.train_img_df1 = self.getdf(self.train_annos_json1, "images")
        self.train_img_df2 = self.getdf(self.train_annos_json2, "images")
        
        self.val_img_df1 = self.getdf(self.val_annos_json1, "images")
        self.val_img_df2 = self.getdf(self.val_annos_json2, "images")
        
        self.train_anno_df1 = self.getdf(self.train_annos_json1, "annotations")
        self.train_anno_df2 = self.getdf(self.train_annos_json2, "annotations")

        self.val_anno_df1 = self.getdf(self.val_annos_json1, "annotations")
        self.val_anno_df2 = self.getdf(self.val_annos_json2, "annotations")
        
    def read_anno_json(self, path):
        with open(path) as file:
            json_dic = json.load(file)
        return json_dic
    
    def getdf(self, dic, key):
        df = pd.DataFrame(dic[key])
        return df
    
    def combine_annos(self):
        max_train_img_id = self.train_img_df1["id"].max()+1
        max_val_img_id = self.val_img_df1["id"].max()+1
        max_train_anno_id = self.train_anno_df1["id"].max()+1
        max_val_anno_id = self.val_anno_df1["id"].max()+1
        
        self.train_img_df2["id"]+= max_train_img_id
        self.val_img_df2["id"]+= max_train_img_id
        
        self.train_anno_df2["image_id"]+= max_train_img_id
        self.val_anno_df2["image_id"]+= max_train_img_id
        self.train_anno_df2["id"]+= max_train_anno_id
        self.val_anno_df2["id"]+= max_val_anno_id
        
        self.train_img_df = pd.concat([self.train_img_df1, self.train_img_df2], ignore_index = True)
        self.val_img_df = pd.concat([self.val_img_df1, self.val_img_df2], ignore_index = True)
        
        self.train_anno_df = pd.concat([self.train_anno_df1, self.train_anno_df2], ignore_index = True)
        self.val_anno_df = pd.concat([self.val_anno_df1, self.val_anno_df2], ignore_index = True)
        
        self.train_img_json = self.train_img_df.to_json(orient = "records")
        self.val_img_json = self.val_img_df.to_json(orient = "records")
        
        self.train_anno_json = self.train_anno_df.to_json(orient = "records")
        self.val_anno_json = self.val_anno_df.to_json(orient = "records")
        
        return
    
    def __create_infos(self):
        #Infoteil
        self.info_dic={"year":datetime.datetime.today().year,\
        "version":"",\
        "description":f"Annotations combined from two COCO Style Datasets. Old Project_Names {self.ds1} {self.ds2} New Project Name : {self.ds_name}",\
        "contributor":"", "url":"", "date_created":datetime.datetime.today()}
        self.info_json=json.dumps(self.info_dic, default=str) 
        return
    
    def __create_licenses(self):
        self.licenses_json=json.dumps(self.train_annos_json1["licenses"])
        return
    
    def __create_categories(self):
        self.cat_json=json.dumps(self.train_annos_json1["categories"])
        return
    
    def do_pipeline(self):
        self.__create_infos()
        self.__create_licenses()
        self.__create_categories()
        self.combine_annos()
        self.__create_temp_dir()
        self.create_full_json(self.train_img_json, self.train_anno_json, "instances_train.json")
        self.create_full_json(self.val_img_json, self.val_anno_json, "instances_val.json")
        self.__copy_images(self.train_img_df1, self.val_img_df1, self.ds1)
        self.__copy_images(self.train_img_df2, self.val_img_df2, self.ds2)
        temp_coco_creator = coco_creator(self.args)
        temp_coco_creator.do_pipeline()
        self.__cleanup()
        return
    
    def __cleanup(self):
        rmtree(self.args.ds_name)
        return
    
    def __create_temp_dir(self):
        os.mkdir(self.ds_name)
        os.mkdir(self.ds_name+ "/annotations")
        os.mkdir(self.ds_name+ "/train")
        os.mkdir(self.ds_name+ "/val")
        return
    
    def create_full_json(self, img_json, anno_json, file_name):
        full_json = {"info": json.loads(self.info_json), \
        "licenses" :   json.loads(self.licenses_json),\
        "categories":  json.loads(self.cat_json)  , \
        "images":  json.loads(img_json)  , \
        "annotations" :  json.loads(anno_json)  , \
                    }
        with open(self.ds_name+"/annotations/"+file_name, "w")\
        as outfile:
            json.dump(full_json, outfile)
        return
    
    def __copy_images(self, train, val, ds):
        for idx, file_name in tqdm(train["file_name"].iteritems()):
                copy2("Yet-Another-EfficientDet-Pytorch/datasets/"+ds+"/train/"+file_name, self.ds_name + "/train/")
        for idx, file_name in tqdm(val["file_name"].iteritems()):
                copy2("Yet-Another-EfficientDet-Pytorch/datasets/"+ds+"/val/"+file_name, self.ds_name + "/val/")
        return
        
        
def get_inputs():
    #Initialize parser
    parser = argparse.ArgumentParser(description='Script to combine two datasets in EfficientDet Repo')
    parser.add_argument("--ds1", action="store", dest = "ds1", type=str, default = "deepfish")
    parser.add_argument("--ds2", action="store", dest = "ds2", type=str, default = None)
    parser.add_argument("--c", action="store", dest = "c", type=int, default = None)
    parser.add_argument("--ds_name", action="store", dest ="ds_name", type=str, default = None)
    #Return parsed object
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = get_inputs()
    myCombiner = combiner(parsed_args)
    myCombiner.do_pipeline()