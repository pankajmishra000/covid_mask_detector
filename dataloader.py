# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:59:59 2020

@author: pankaj.mishra
"""
# load the wonders (libraries)
import torch.utils.data as data
import torch
from torchvision import transforms

import skimage.io as io
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# read images
def load_images(path, image_name):
    return io.imread(os.path.join(path,image_name))

def load_train_file(path):
    '''
    Args:
        path : Path of the base folder './data'
        
    Return: Return train images and train lables 
    '''
    train_images = []
    train_labels = []
    files = os.listdir(path)
    for file in files:
        
        for n,i in enumerate(tqdm(os.listdir(os.path.join(path,file)))):
            image = load_images(os.path.join(path,file),i)
            train_images.append(image)
            if file == "with_mask":
                train_labels.append(np.ones(1))
            else:
                train_labels.append(np.zeros(1))
        print(f'\nTotal {file} images: {n}\n')
    return tuple(zip(train_images, train_labels))


class MaskLoader:
    def __init__(self, path='D:\\Python\\Codes\\tensorflow codes\\Mask_Detection\\data', batch_size =4):
        self.path = path
        self.batch= batch_size
        
        # Image Transformation ##
        T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((256)),
            transforms.CenterCrop(224),            
            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,)),
            ])
        print('------ Loading Images -----\n')
        img_labl = load_train_file(self.path)
        train, test = train_test_split(img_labl, test_size=0.2, random_state=131)
        
        print('\n**** Applying Transformations over Train Dataset ****')
        train_img = torch.stack([T(train[i][0]) for i in tqdm(range(len(train)))])
        print('\n**** Applying Transformations over Validation Dataset ****\n')
        test_img = torch.stack([T(test[i][0]) for i in tqdm(range(len(test)))])
        
        train_labl = torch.stack([torch.tensor(train[i][1], dtype = torch.long) for i in range(len(train))])
        test_label = torch.stack([torch.tensor(test[i][1], dtype = torch.long) for i in range(len(test))])
        
        train_f = tuple(zip(train_img, train_labl))
        test_f = tuple(zip(test_img, test_label))
        
        self.train_loader = torch.utils.data.DataLoader(train_f, batch_size=self.batch, pin_memory=True, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_f, batch_size=self.batch, pin_memory=True, shuffle=False)
        



if __name__ == "__main__":
    path = "D:\\Python\\Codes\\tensorflow codes\\Mask_Detection\\data"
    data = MaskLoader(batch_size = 6)
    for img, label in data.train_loader:
        print(f'size of image: {img.shape}')
        plt.imshow(img[0].permute(1,2,0).cpu().numpy())
        plt.show()
        break