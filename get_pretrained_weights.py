import urllib.request
import os
import sys

if __name__ == "__main__":

    folder = "Yet-Another-EfficientDet-Pytorch/weights/"
    if "weights" not in os.listdir("Yet-Another-EfficientDet-Pytorch/"):
        os.mkdir(folder)

    urls = ['https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth', "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth", "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth"]

    for url in urls:
        urllib.request.urlretrieve(url, filename = folder+url.split("/")[-1])

    urllib.request.urlretrieve("https://drive.google.com/uc?id=1VV60BNwRXGl3G1kX3sJfAImI7BC55sPD&export=download", filename = folder + "FishDet.pth")
    
    
    