# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:01:39 2020

@author: pankaj.mishra
"""

import torch
import torch.nn as nn
import model as md
import dataloader
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
batch_size = 100
epoch = 100

model = md.MaskDetector().cuda()
model.train()

data = dataloader.MaskLoader(batch_size=batch_size)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # Write a code for weight decaying scheme

# Criteria
criterion = nn.CrossEntropyLoss()

# Training
for i in range(epoch):
    loss_epoch = []
    for img, label in data.train_loader:
        img = img.cuda()
        label = label.cuda().squeeze(1)
        
        model.zero_grad()
        output = model(img)
        loss = criterion(output,label)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        
    print(f'loss of epoch {i}: {np.mean(loss_epoch):.2f}')
writer.close()
# Saving last trained model    
torch.save(model.state_dict(), f'model0'+'.pt')


