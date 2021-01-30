import subprocess
from shutil import copy2

def os_command(cmd): 
        print("Executing: "+cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, text=True, \
                            encoding = "utf8")
        while True:
            inchar = p.stdout.read(1)
            if inchar: #neither empty string nor None
                print(str(inchar), end='') #or end=None to flush immediately
            else:
                print('') #flush for implicit line-buffering
                break
        return
    
os_command("git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch")
copy2("util_files/infer.py", "Yet-Another-EfficientDet-Pytorch/infer.py")
os_command("conda env create -f util_files/environment.yml")

#Now run /pathtomyenvs/envs/FishDet/bin/python Interface.py to use