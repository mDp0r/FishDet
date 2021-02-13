import os
import datetime
import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tqdm import tqdm
from shutil import copy2

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=str, default='coco', help='project file that contains parameters')
    ap.add_argument('--c',  type=int, default=0, help='coefficients of efficientdet')
    ap.add_argument('--w',  type=str, default=None, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--cuda', type=boolean_string, default=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=boolean_string, default=False)
    ap.add_argument('--mode', type=str, default='all', help='viz:visualize pics, coco:create coco annos, lost: create lost annos, all:does everything')
    ap.add_argument('--path', type=str, default=None, help='Path to imgs in Yet-Another-EfficientDet Folder')
    ap.add_argument('--conf_threshold', type=float, default=0.05, help="Filters BBoxes with Confidence smaller this value")
    #ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
    return ap.parse_args()

class FishPred:
    
    def __init__(self, args):
        self.p = args.p
        self.c = args.c
        self.weights = args.w
        self.nms_threshold = args.nms_threshold
        self.cuda = args.cuda
        self.float16 = args.float16
        self.device = args.device
        self.mode = args.mode
        self.session = self.__create_dirs()
        self.impath = args.path if args.path else "datasets/"+ args.p +"/val/"
        self.input_sizes =[512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.p_params = yaml.safe_load(open('projects/'+self.p+".yml"))
        self.conf_threshold = args.conf_threshold
        self.__save_configs()
        
        return
    
    def __create_dirs(self):
        print("Creating directories")
        if "inference" not in os.listdir():
            os.mkdir("inference")
        session_path = "inference/"+datetime.datetime.today().strftime("%Y%m%d_%H%M%S")+"/"
        os.mkdir(session_path)
        
        if (self.mode == "all") or (self.mode == "viz"):
            os.mkdir(session_path+"imgs_with_bboxes")
            
        if (self.mode == "all") or (self.mode == "coco"):
            os.mkdir(session_path+"coco_annos/")
            os.mkdir(session_path+"coco_annos/annotations/")
            os.mkdir(session_path+"coco_annos/train/")
            os.mkdir(session_path+"coco_annos/val/")
            
        return session_path
    
    
    
    def __save_configs(self):
        print("Writing Config file")
        with open(self.session+"Configs.txt", "w") as text_file:
            print(f"Weights: {self.weights}\nProject: {self.p}\nImgpath: {self.impath}\nmode: {self.mode}", file=text_file)
        
        
        return
    
    def do_command(self):
        print("Starting predictions....")
        self.__predict()
        print("Mode "+self.mode+" seletected")
        if self.mode == "all":
            print("Writing COCO Annos")
            self.__write_coco()
            print("Writing LOST Annos")
            self.__write_lost()
            print("Saving Images with Bboxes")
            self.__viz()
            
        elif self.mode =="coco":
            print("Writing COCO Annos")
            self.__write_coco()
            
        elif self.mode =="lost":
            print("Writing LOST Annos")
            self.__write_lost()
            
        elif self.mode == "viz":
            print("Saving Images with Bboxes")
            self.__viz()
        else:
            print("Error. No valid mode specified")
        return
    
    def __predict(self):
        
        
        
        model = EfficientDetBackbone(compound_coef=self.c, num_classes=len(self.p_params["obj_list"]),
                                         ratios=eval(self.p_params['anchors_ratios']), scales=eval(self.p_params['anchors_scales']))
            
        model.load_state_dict(torch.load(self.weights, map_location=torch.device('cpu')))

        model.requires_grad_(False)
        if self.cuda:
            model.cuda(self.device)
            if self.float16:
                model.half()
        model.eval()
        imgs = [self.impath+x for x in os.listdir(self.impath)]
        results = []

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        for image in tqdm(imgs):

            ori_imgs, framed_imgs, framed_metas = preprocess(image, max_size=self.input_sizes[self.c])
            x = torch.from_numpy(framed_imgs[0])

            if self.cuda:
                x = x.cuda(self.device)
                if self.float16:
                    x = x.half()
                else:
                    x = x.float()
            else:
                x = x.float()

            x = x.unsqueeze(0).permute(0, 3, 1, 2)
            features, regression, classification, anchors = model(x)

            preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                self.conf_threshold, self.nms_threshold)
            if not preds:
                continue

            preds = invert_affine(framed_metas, preds)[0]

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            if rois.shape[0] > 0:
                # x1,y1,x2,y2 -> x1,y1,w,h
                rois[:, 2] -= rois[:, 0]
                rois[:, 3] -= rois[:, 1]

                bbox_score = scores

                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    label = int(class_ids[roi_id])
                    box = rois[roi_id, :]

                    image_result = {
                        'img_pth': image,
                        'category_id': label + 1,
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)
        
        self.results = pd.DataFrame(results)
        self.results["bbox"] =self.results["bbox"].apply\
    (lambda x: np.rint(x,np.zeros(4,int),casting='unsafe'))
        self.results ["x"] = self.results["bbox"].apply(lambda x : x[0])
        self.results ["y"] = self.results["bbox"].apply(lambda x : x[1])
        self.results ["width"] = self.results["bbox"].apply(lambda x : x[2])
        self.results["height"] = self.results["bbox"].apply(lambda x : x[2])
       
    def __viz(self):
        for temp_path in tqdm(self.results["img_pth"].unique()):
            self.__viz_annos_per_img(temp_path, self.results.copy().loc[self.results["img_pth"]==temp_path], self.session+"imgs_with_bboxes/"+temp_path.split("/")[-1])
        return
    def __viz_annos_per_img(self,img_path, anno_df, target_path):
        im = np.array(Image.open(img_path))
        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        for i,anno in anno_df.iterrows():
            # Create a Rectangle patch
            rect = patches.Rectangle((anno["x"],anno["y"]),anno["width"],anno["height"],linewidth=1,edgecolor='g',facecolor='none', label =str(anno["score"]))
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(anno["x"], anno["y"], f'({int(np.around(anno["score"], decimals=2)*100)})', color='r', fontsize = 4)
        #plt.show()
        plt.savefig(target_path, dpi=250)
        plt.close()
        return
    
    def __write_coco(self):
        self.__create_infos()
        self.__create_licenses()
        self.__create_categories()
        self.__train_val_split()
        self.train_img_df, self.train_anno_df, self.train_img_json,\
        self.train_anno_json = self.__get_annos_and_img_metadata(self.__train)
        
        self.val_img_df, self.val_anno_df, self.val_img_json,\
        self.val_anno_json = self.__get_annos_and_img_metadata(self.__val)
        
        self.__create_full_json(self.info_json, self.licenses_json, self.cat_json,\
        self.train_img_json,self.train_anno_json, "instances_train.json")

        self.__create_full_json(self.info_json, self.licenses_json, self.cat_json,\
        self.val_img_json,self.val_anno_json, "instances_val.json")

        return
    
    def __train_val_split(self):
        imgs = self.results["img_pth"].unique()
        train_imgs = imgs[:int(np.around(len(imgs)/2))]
        
        self.__train = self.results.loc\
        [self.results["img_pth"].isin(train_imgs)].copy()
        
        self.__val = self.results.loc\
        [~self.results["img_pth"].isin(train_imgs)].copy()
        
        for pth in tqdm(imgs):
            if pth in train_imgs:
                copy2(pth, self.session+"coco_annos/train/")
            else:
                copy2(pth, self.session+"coco_annos/val/")
        return
    
    def __create_infos(self):
        #Infoteil
        self.info_dic={"year":datetime.datetime.today().year,\
        "version":"",\
        "description":f"Auto Generated Annotations. Project_Name {self.p}",\
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
        for x in self.p_params["obj_list"]:    
            self.cat_list.append({"id":self.p_params["obj_list"].index(x)+1\
                             , "name":x, "supercategory":""})
        self.cat_json=json.dumps(self.cat_list)
        return
    
   
    
    def __generate_img_metadata(self, df):
        df = df.drop(columns = \
        ["score", "x","y","width","height","bbox","category_id"])
        df = df.drop_duplicates()
        df["file_name"] = df["img_pth"].str.split("/").str[-1]
        df["width"] = df["img_pth"].apply(lambda x: Image.open(x).width)
        df["height"] = df["img_pth"].apply(lambda x: Image.open(x).height)
        df["license"] = 1
        df = df.reset_index(drop=True).reset_index().rename(columns={"index":"id"})
        return df
    
    def __process_annos(self,df):
        df["area"] = df["width"] * df["height"]
        df = df.drop(columns=["x","y","width","height","score"])
        df["iscrowd"] = 0
        df["segmentation"]=np.empty((len(df), 0)).tolist()
        df = df.reset_index(drop=True).reset_index().rename(columns={"index":"id"})
        return df
    
    def __replace_pth_with_id(self,imdf, annodf):
        annodf = annodf\
        .merge(imdf.rename(columns={"id":"image_id"})\
        .loc[:,["img_pth","image_id"]], how="left", on=["img_pth"])\
        .drop(columns=["img_pth"])
        imdf = imdf.drop(columns=["img_pth"])
        return imdf, annodf
    
    def __get_annos_and_img_metadata(self,data):
        imgs_df = self.__generate_img_metadata(data)
        anno_df = self.__process_annos(data)
        imgs_df, anno_df = self.__replace_pth_with_id(imgs_df, anno_df)
        return imgs_df, anno_df, imgs_df.to_json(orient="records"),\
        anno_df.to_json(orient="records")
    
    def __create_full_json(self,info, licenses, cat, images, annos, filename):
        full_json = {"info": json.loads(info), \
        "licenses" :   json.loads(licenses),\
        "categories":  json.loads(cat)  , \
        "images":  json.loads(images)  , \
        "annotations" :  json.loads(annos)  , \
                    }
        with open(self.session+"coco_annos/annotations/"+filename, "w")\
        as outfile:
            json.dump(full_json, outfile)
        return
    
    def __write_lost(self):
        self.lost_annos = self.__format_bbox_lost(self.results)
        self.lost_annos = self.__format_output_lost(self.lost_annos)
        self.__write_output_lost(self.lost_annos)
        
        return
    
    def __format_bbox_lost(self, annos):
        annos["x"] = annos["x"].copy()+0.5*annos["width"].copy()
        annos["y"] = annos["y"].copy()+0.5*annos["height"].copy()
        annos["im_width"] = annos["img_pth"].apply(lambda x: Image.open(x).width)
        annos["im_height"] = annos["img_pth"].apply(lambda x: Image.open(x).height)
        annos["x"] = annos["x"].copy() / annos["im_width"].copy()
        annos["width"] = annos["width"].copy() / annos["im_width"].copy()
        annos["y"] = annos["y"].copy() / annos["im_height"].copy()
        annos["height"] = annos["height"].copy() / annos["im_height"].copy()
        annos["anno.data"] = annos.apply(lambda x:json.dumps({'x':x["x"], 'y':x["y"], 'w':x["width"], 'h':x["height"]}) , axis=1)
        return annos
    
    def __format_output_lost(self,annos):
        annos = annos.copy().loc[:,["anno.data", "category_id", "img_pth"]]
        annos = annos.copy().rename(columns = {"category_id":"anno.lbl.external_id", "img_pth":"img.img_path"})
        annos["anno.lbl.name"] = annos["anno.lbl.external_id"].apply(lambda x: json.dumps([self.p_params["obj_list"][x-1]]))
        annos["anno.lbl.external_id"] = annos["anno.lbl.external_id"].apply(lambda x: json.dumps([str(x)]))
        return annos
    
    def __write_output_lost(self, annos):
        annos.to_csv(self.session+"lost_annos.csv", index=False)
        return
        
    
    
if __name__ == "__main__":
    args = parse_arguments()
    myFishPred = FishPred(args)
    myFishPred.do_command()