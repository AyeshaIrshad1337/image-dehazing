import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def preprocess_img(img_path, size):
    img=cv2.imread(img_path)
    img=cv2.resize(img, (size, size))
    img=img/255
    return img
class Dehaze_Image_Dataset:
    def __init__ (self, input_dir, target_dir, size):
        self.input_dir=input_dir
        self.target_dir=target_dir
        self.size=size
        self.input_imgs=os.listdir(input_dir)
        self.target_imgs=os.listdir(target_dir)
    def __len__(self):
        return len(self.input_imgs)
    def __getitem__(self, idx):
        input_img_path=os.path.join(self.input_dir,self.input_imgs[idx])
        target_img_path=os.path.join(self.target_dir,self.target_imgs[idx])

        input_ing=preprocess_img(input_img_path, self.size)
        target_img=preprocess_img(target_img_path, self.size)

        input_img=torch.tensor(input_img, dtype=torch.float32).permute(2,0,1)
        target_img=torch.tensor(target_img, dtype=torch.float32).permute(2,0,1)

        return input_img, target_img