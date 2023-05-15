# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:57:28 2023

@author: Zaifu Zhan
"""

from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
# print(dir(models))

resnet = models.resnet101(pretrained=True)
# print(resnet)

# load image
img = Image.open('dog.png')

# def preprocess
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                  transforms.ToTensor(), transforms.Normalize(mean = [0.485,0.456, 0.406], std = [0.229, 0.224, 0.225])])
img_t = preprocess(img)


batch_t = torch.unsqueeze(img_t,0)
resnet.eval()
out = resnet(batch_t)


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

print('The picture is no.' + labels[index[0]])