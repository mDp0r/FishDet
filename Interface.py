#ToDo
#Add Infer part
#Add cleanup (copy latest model, remove checkpoints)
#Kleinen kroatischen Datensatz Annos vorbereiten und Viz testen. Auch kmeans laufen lassen für Anchors

#Imports
import os
import subprocess
import argparse
import sys
#Main Class
class FishDet:
    """
    Initialisation of Class 
    _______________________
    Input Arguments:
    _______________________
    args, type namespace, contains parsed arguments, includes following attributes
        do: type string, only train, eval, infer supported
        batch_size: Type int /needed for train
        lr: type float, is "learnrate" /needed for train
        num_epochs: Type int /needed for train
        detector: Type String, So far only EfficientDet supported
        c: Compound param of EfficientDet, should be None if not EfficientDet
        project: Type String, See folder name in /datasets and filename of yaml in projects of EfficientDet Pytorch so far,
        load_weights: Type String, Default None, otherwise name of .pth File
        head_only: Type Bool, default True
        optim: Type String, default adamw, sgd also supported /used for train
        debug: Type Bool, default False. Used in EfficientDet Vizualisation to draw boxes during training /used for train
        infer_mode: Type String, default "all", so far "all", "coco" and "viz" supported /needed for inference
        conf_threshold : Type Float, default 0.05 / Threshold to Filter Bboxes in the postprocessing after predicting / needed for inference
        path: Type String, default None /needed for inference
    _______________________
    Function:
    _______________________
    Initializes attributes with Inputs, also calls detector specific init functions.
    """  
    def __init__ (self, args):
        
        #Set Input as attributes
        self.do = args.do
        self.detector = args.detector
        self.project = args.project
        self.load_weights = args.load_weights
        self.project = args.project
        self.head_only = args.head_only
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.optim = args.optim
        self.debug = args.debug
        self.conf_threshold = args.conf_threshold
        self.infer_mode = args.infer_mode
        self.path = args.path
        #Do Initialisation based on detector
        if self.detector == "EfficientDet":
            self.__init_EfficientDet(args.c)
        
        #elif self.detector == "CSPNet":
            #self.__init_CSPNet()
    
        return
    
    """
    Initialization of EfficientDet
    _________
    Function:
    _________
    Does EfficientDet specific Initialisation. So far just cds into right dir of
    EfficientDet implementation and saves compound param to object attribute
    
    """
    def __init_EfficientDet(self, c):
        
        os.chdir("Yet-Another-EfficientDet-Pytorch")
        self.c = c
        return
    
    """
    Does command according to do attribute.
    _________
    Function:
    _________
    Can be called outside of class. Does appropriate command. So far just "train" and "eval".
    specified.
    
    """
    def do_command(self):
        if self.do == "train":
            self.__train()
        if self.do == "eval":
            self.__eval()
        elif self.do == "infer":
            self.__infer()
        return
        
    
    """
    train method:
    _____________
    Function:
    _____________
    Depending on Detector calls right train method
    
    """
    def __train(self):
        
        if self.detector == "EfficientDet":
            self.__train_EfficientDet()
            
            
        #elif self.detector == "CSPNet"
            #self.__train_CSPNet()
            
        return
    
    """
    train Interface function for EfficientDet
    _________
    Function:
    _________
    Creates Shell Command String from object attributes. Spawns subprocess with 
    with command.
    
    """
    
    def __train_EfficientDet(self):
        self.__os_command(self.__get_EfficientDet_train_cmd())
        return    
    
    
    """
    Shell Command Creation for EfficientDet Training
    ___________
    Function:
    ___________
    Creates Shell Command String from Inputs specified at Object Creation.
    ___________
    Outputs:
    ___________
    cmd, type String
    
    """
    def __get_EfficientDet_train_cmd(self):
        
        cmd = "python -u train.py"

        # training with preset weights
        if self.load_weights is not None:
            cmd += " --load_weights weights/" + self.load_weights

        #Add head only option to cmd string
        if self.head_only is not None:
            cmd += " --head_only " + str(self.head_only)

        #Add optim to cmd string
        if self.optim is not None:
            cmd += " --optim " + str(self.optim)
        
        #Add debug option to cmd string
        if self.debug is not None:
            cmd+= " --debug " + str(self.debug)
            
        #Add rest of arguments
        cmd += " --batch_size " + str(self.batch_size) + " --lr " + str(self.lr) + \
        " --num_epochs " + str(self.num_epochs) + " -p " + str(self.project) + \
        " --num_workers 0" + " -c " + str(self.c) \
        + "--save_interval " + str(sys.maxsize) 
    
        return cmd
        
    """
    Function for spawning subprocess with command
    _________
    Inputs:
    _________
    cmd: Type String, command you would enter in console
    _________
    Function:
    _________
    Uses subprocess library to spawn new subprocess on os with command.
    Also prints in the executing console the Output of the executed process.
    
    """
    def __os_command(self, cmd): 
        print("Command used in subprocess: "+cmd)
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
    
        """
    eval method:
    _____________
    Function:
    _____________
    Depending on Detector calls right eval method
    
    """
    def __eval(self):
        
        if self.detector == "EfficientDet":
            self.__eval_EfficientDet()
            
            
        #elif self.detector == "CSPNet"
            #self.__eval_CSPNet()
            
        return
    
    """
    Evaluation function for EfficientDet
    __________
    Function:
    __________
    Uses get_EfficientDet_eval_command method to get the right eval command considering Input attributes. Then calls __os_command function
    to spawn subprocess with function.
    
    """
    def __eval_EfficientDet(self):
        self.__os_command(self.__get_EfficientDet_eval_command())
        return
    
    """
    Function to create coco eval command for EfficientDet Repo. 
    _________
    Function:
    Appends input parameters according to specified attributes. Will be used to call in os_command.
    _________
    Output:
    _________
    Type: String
    
    """
    def __get_EfficientDet_eval_command(self): 
        return "python -u coco_eval.py" + " -c " + str(self.c)+ " -p " + str(self.project) + " -w weights/" + str(self.load_weights)
    
    """
    infer method:
    _____________
    Function:
    _____________
    Depending on Detector calls right train method
    
    """
    def __infer(self):
        
        if self.detector == "EfficientDet":
            self.__infer_EfficientDet()
            
            
        #elif self.detector == "CSPNet"
            #self.__infer_CSPNet()
            
        return
    
    """
    Infer function for EfficientDet
    __________
    Function:
    __________
    Uses get_EfficientDet_infer_command method to get the right eval command considering Input attributes. Then calls __os_command function
    to spawn subprocess with function.
    
    """
    def __infer_EfficientDet(self):
        self.__os_command(self.__get_EfficientDet_infer_command())
        return
    
    """
    Function to create infer command for EfficientDet Repo. 
    _________
    Function:
    Appends input parameters according to specified attributes. Will be used to call in os_command.
    _________
    Output:
    _________
    Type: String
    
    """
    def __get_EfficientDet_infer_command(self): 
        return "python -u infer.py" + " --c " + str(self.c)+ " --p " + str(self.project) + " --w weights/" + str(self.load_weights) + \
    " --mode " + str(self.infer_mode) + " --path " + str(self.path) + " --conf_threshold " + str(self.conf_threshold)
    
    
#Other functions

"""
Function to get inputs from command line
_________
Function:
_________
Creates and specifies parser for this script. Parses Items.
_________
Output:
________
Type: namespace
"""
def get_inputs():
    #Initialize parser
    parser = argparse.ArgumentParser(description='Interface Script for Fish Detection')
    #Add Arguments
    parser.add_argument("--do", action="store", dest = "do", default = None)
    parser.add_argument("--batch_size", action="store", dest = "batch_size", type=int, \
default = None)
    parser.add_argument("--lr", action="store", dest = "lr", type=float, default = None)
    parser.add_argument("--num_epochs", action="store", dest = "num_epochs", type=int,\
default = None)
    parser.add_argument("--detector", action="store", dest = "detector", default = "EfficientDet")
    parser.add_argument("--c", action="store", dest = "c", type=int, default = None)
    parser.add_argument("--project", action="store", dest = "project", default = None)
    parser.add_argument("--load_weights", action="store", dest = "load_weights", \
default = None)
    parser.add_argument("--head_only", action="store", dest="head_only", type=bool, \
default = None)
    parser.add_argument("--optim", action="store", dest="optim", type=str, \
default = None)
    parser.add_argument("--debug", action="store", dest="debug", type=bool, \
default = None)
    parser.add_argument("--infer_mode", action="store", dest="infer_mode", type=str, \
default = "all")
    parser.add_argument("--path", action="store", dest="path", type=str, \
default = None)
    parser.add_argument ("--conf_threshold", action = "store", dest ="conf_threshold", type=float, default = 0.05)
    #Return parsed object
    return parser.parse_args()

if __name__ == "__main__":
    #Parse inputs
    inputs=get_inputs()
    #Create Object
    myFishDet = FishDet(inputs)
    #Do Command
    myFishDet.do_command()
    