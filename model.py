# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:35:55 2020

@author: pankaj.mishra
"""

import torch
import torchvision
import dataloader
import torch.nn as nn
import torch.nn.functional as F

class MaskDetector(nn.Module):
    def __init__(self,):
        super(MaskDetector, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model.classifier[-1] = nn.Linear(1280,2)
        
    def forward(self, x):
        prob = self.model(x)
        return F.softmax(prob)
    
if __name__ == "__main__":
    from torchsummary import summary
    model = MaskDetector().cuda()
    print(model)
    model = model.cuda()
    summary(model, input_size = (3,224,224))
