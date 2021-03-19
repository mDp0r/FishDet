"""

def download_model(s):
    model_url = s.get_arg('model_url')
    model_path = s.get_path(os.path.basename(model_url), context='static')
    if os.path.exists(model_path):
        return model_path
    s.logger.info('Download yolo model from: {}'.format(model_url))
    with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    s.logger.info('Stored model file in static context: {}'.format(model_path))
    return model_path

ToDO:
Nochmal schauen wie das mit Daten in Lost ist
Dann: Alle Annos einlesen
Je Bild filtern
Und request mit API stellen

"""
from lost.pyapi import script
import os
import random
import pandas as pd
import datetime 
import time 

ENVS = ['lost']

ARGUMENTS = {'polygon' : { 'value':'false',
                            'help': 'Add a dummy polygon proposal as example.'},
            'line' : { 'value':'false',
                            'help': 'Add a dummy line proposal as example.'},
            'point' : { 'value':'false',
                            'help': 'Add a dummy point proposal as example.'},
            'bbox' : { 'value':'true',
                            'help': 'Add a dummy bbox proposal as example.'}
            }

def extract_bbox_as_list(anno):
    return [ eval(anno) [x] for x in eval(anno)]

def get_latest(path):
    csvs = [x for x in os.listdir(path) if x.find(".csv")!= -1]
    csvs.sort()
    return path + "/" +csvs[-1]
def get_timestamp(path_string):
    return datetime.datetime.strptime(path_string[-14:-4], "%y%m%d%H%M")


class RequestAnnos(script.Script):
    '''Request annotations for each image of an imageset.

    An imageset is basicly a folder with images.
    '''
    def main(self):
        media_path = "/home/lost/data/media/fishes/"
        fishdet_path = "/home/lost/FishDet/"
                
        tracking_path = fishdet_path+"anno_in_tracking.txt"
        anno_dir_path = fishdet_path+"annos_in/"

        latest_anno_path = get_latest(anno_dir_path)
        latest_timestamp = get_timestamp(latest_anno_path)



        if os.path.exists(tracking_path):
            with open (tracking_path) as file:
                last_processed = file.read()
            last_processed_timestamp = get_timestamp(last_processed)
        else:
            last_processed_timestamp = None

        if (last_processed_timestamp) and (last_processed_timestamp>=latest_timestamp):
            #Sleep and kill script
            self.logger.info('Waiting for new annotations.')
            time.sleep(60)
            self.reject_execution()
        else:
            annos = pd.read_csv(latest_anno_path)
            annos["img.file_name"] = annos["img.img_path"].str.split("/").str[-1]
            for img_file in os.listdir(media_path):
                img_path = media_path + img_file
                temp_annos = annos.copy().loc[annos["img.file_name"] == img_file]
                anno_types = []
                anno_list = []
                anno_labels = []
                labels_p = self.outp.sia_tasks[0].possible_label_df.copy().set_index("external_id")
                label = labels_p.at["1", "idx"]
                for i, row in temp_annos.iterrows():
                    anno_list.append(extract_bbox_as_list(row["anno.data"]))
                    anno_types.append('bbox')
                    anno_labels.append([label])
                                            
                self.outp.request_annos(img_path=img_path, annos=anno_list, anno_types=anno_types, anno_labels = anno_labels)
                self.logger.info('Requested annos for: {}'.format(img_path))
            with open (tracking_path, "w") as file:
                file.write(latest_anno_path)
    
    
    
        
        
        
        

if __name__ == "__main__":
    my_script = RequestAnnos() 
