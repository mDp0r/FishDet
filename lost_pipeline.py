import os
import subprocess
from shutil import copy2, copytree
import argparse
import sys
import datetime
import time
import pathlib

def os_command(cmd):
    print("Executing: "+cmd)
    p = subprocess.call(cmd, shell = True)
    return
    

def get_latest(path):
    csvs = [x for x in os.listdir(path) if x.find(".csv")!= -1]
    csvs.sort()
    return path + "/" +csvs[-1]

def get_timestamp(path_string):
    return datetime.datetime.strptime(path_string[-14:-4], "%y%m%d%H%M")

def get_creation_tmstmp(file_path):
    fname = pathlib.Path(file_path)
    ctime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
    return ctime


def parse_args():
    #Initialize parser
    parser = argparse.ArgumentParser(description='Script to setup Dataset in EfficientDet Repo')
    parser.add_argument("--c", action="store", dest = "c", type=int, default = None)
    parser.add_argument("--path", action="store", dest="path", type=str, \
default = None)
    #Return parsed object
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    if not os.path.exists("Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d4.pth"):
        print("COCO pretrained weights for EfficientDet D4 not found. Downloading pretrained weights.")
        os_command("python get_pretrained_weights.py")
    if not os.path.exists("Yet-Another-EfficientDet-Pytorch/projects/deepfish.yml"):
        print("Deepfish dataset not found. Downloading and integrating into EfficientDet.")
        os_command("python setup_ds.py --mode known --ds_name deepfish")
    fishdet_path = args.path + "lost/data/FishDet/"
    if not os.path.exists(fishdet_path):
        print("FishDet Path in Lost doesn't exist. Doing rudimentary setup from util_files. You will need to copy images \
              /path/to/lost/lost/data/FishDet/imgs/. In the Docker container you will also need to copy the images to \
              /home/lost/data/media/fishes/. You also need to integrate the pipeline(fishdet) and the label tree (tree.csv) into\
              lost. See lost.readthedocs.io for more information.")
        copytree("util_files/lost_project/FishDet/", fishdet_path)
        os.mkdir(fishdet_path + "annos_in")
        os.mkdir(fishdet_path + "annos_out")
        os.mkdir(fishdet_path + "imgs")
        print("Did Setup. Now aborting. Run again when above described ToDos are made.")
        sys.exit()

    if len(os.listdir(fishdet_path+"imgs")) == 0:
        print("imgs folder still empty. Put images /path/to/lost/lost/data/FishDet/imgs/")
        sys.exit()

    if "fishes" not in os.listdir(args.path + "/lost/data/data/media"):
        print("You need to also copy the images in the container to /home/lost/data/media/fishes !")
        sys.exit()
    if len(os.listdir(args.path + "/lost/data/data/media/fishes"))== 0:
        print("You need to also copy the images in the container to /home/lost/data/media/fishes !")
        sys.exit()

    tracking_path = fishdet_path+"anno_out_tracking.txt"
    anno_dir_path = fishdet_path+"annos_out/"

    


    while(True):
        latest_anno_path = get_latest(anno_dir_path)
        latest_timestamp = get_timestamp(latest_anno_path)

        if os.path.exists(tracking_path):
            with open (tracking_path) as file:
                last_processed = file.read()
            last_processed_timestamp = get_timestamp(last_processed)
        else:
            last_processed_timestamp = None

        if (last_processed_timestamp) and (last_processed_timestamp>=latest_timestamp):
            #Sleep 
            current_timestamp = datetime.datetime.today().strftime("%y%m%d%H%M")
            print(f"Wating for new annos {current_timestamp}")
            time.sleep(30)
        else:
            #setup lost ds from annos
            ds_name = "fishes_" + datetime.datetime.today().strftime("%y%m%d%H%M")
            ds_name2 = "deepfishes_" + datetime.datetime.today().strftime("%y%m%d%H%M")
            latest_anno_file = latest_anno_path.split("/")[-1]
            print("Setting up coco style dataset from lost annos. Also integrating it into efficientdet. For params see above command")
            os_command(f"python setup_ds.py --mode lost --c 4 --ds_name {ds_name} --anno_file {latest_anno_file} --path {fishdet_path}")
            #combine with deepfish annos
            print("Setting up coco style dataset as combination of deepfish and new fish annos.\
            Also integrating it into efficientdet. For params see above command")
            os_command(f"python combine_ds.py --ds1 deepfish --ds2 {ds_name} --ds_name {ds_name2} --c 4 ")
            #train
            print("Starting training...This might take a while")
            os_command(f"python Interface.py --do train --project {ds_name2} --c 4 --load_weights efficientdet-d4.pth --detector EfficientDet --batch_size 1 --lr 1e-4 --num_epochs 120")
            #copy best model to weights
            print("Copying best model to weights")
            train_weights_path = f"Yet-Another-EfficientDet-Pytorch/logs/{ds_name2}/"
            weights = [x for x in os.listdir(train_weights_path) if ".pth" in x]
            latest_weights_pos = 0
            latest_weights_tmstmp = get_creation_tmstmp(train_weights_path+weights[latest_weights_pos]) 
            for weight_pos in range(len(weights)):
                temp_weights_tmstmp = get_creation_tmstmp(train_weights_path + weights[weight_pos])
                if temp_weights_tmstmp > latest_weights_tmstmp:
                    latest_weights_tmstmp = temp_weights_tmstmp
                    latest_weights_pos = weight_pos
            best_weights_path = train_weights_path + weights[latest_weights_pos]
            name_of_model = "FishDet_" + datetime.datetime.today().strftime("%y%m%d%H%M") +".pth"
            copy2(best_weights_path, "Yet-Another-EfficientDet-Pytorch/weights/"+name_of_model)
            #cleanup
            #Might be added in the future. Shouldnt matter because there is a new project created in each run.
            #infer
            img_path = fishdet_path+"imgs/"
            os_command(f"python Interface.py --do infer --project {ds_name2} --c 4 --load_weights {name_of_model} --detector EfficientDet --infer_mode all --path {img_path} --conf_threshold 0.5")
            #copy lost annos to lost exchange folder
            print("Copying lost annos to lost exchange folder")
            inference_path = "Yet-Another-EfficientDet-Pytorch/inference/" 
            sessions = os.listdir(inference_path)
            sessions.sort()
            lost_annos_path = inference_path + sessions[-1] +"/lost_annos.csv"
            anno_file_name = "annos_" + datetime.datetime.today().strftime("%y%m%d%H%M") +".csv"
            copy2(lost_annos_path, fishdet_path + "annos_in/" + anno_file_name)
            #Write anno tracking
            print("Writing anno tracking")
            with open (tracking_path, "w") as file:
                file.write(latest_anno_path)