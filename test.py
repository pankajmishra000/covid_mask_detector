# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:34:41 2020

@author: pankaj.mishra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import dataloader
from model import MaskDetector
from tqdm import tqdm
import pandas as pd 



model = MaskDetector()

model = model.cuda()
model.load_state_dict(torch.load(f'model0'+'.pt'))
model.eval() 

data = dataloader.MaskLoader(batch_size=1)

with torch.no_grad():
    result =[]
    correct = 0
    for img, label in data.test_loader:
        img = img.cuda()
        label = label.cuda().squeeze(1)
        
        model.zero_grad()
        output = model(img)
        result.append(torch.argmax(output))
        if label ==torch.argmax(output):
            correct +=1
    print(f'Total accuracy: {correct*100/len(result)}')
            
        
