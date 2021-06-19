import time
import picamera
import numpy as np
import os
import psutil
import torchvision.models as models
from torchvision.transforms import transforms
from timeit import default_timer as timer
from datetime import timedelta
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import torch
import torch.nn as nn
from src.model import Model

SOFTMAX = nn.Softmax(dim=1)
CLASSES = ['Battery', 'Clothing', 'Glass', 'Metal', 'Paper', 'Paperpack', 'Plastic', 'Plasticbag', 'Styrofoam']
nSamples = [348, 901, 1558, 2225, 9193, 1944, 5057, 9293, 2080] # ALL TRAIN
class_weight = [1 - (x / sum(nSamples)) for x in nSamples]
class_weight = torch.FloatTensor(class_weight)


def read_yaml(cfg):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config

def get_nas_model():
    model_config = read_yaml(cfg="/home/pi/models/GTSNet.yaml")
    model_instance = Model(model_config, verbose=False)
    net = model_instance.model
    return net

class Inference:
    def __init__(self, model_type = "GTSNet", model_path = None):
        if model_type == "GTSNet":
            self.model = get_nas_model()
        elif model_type == "Shufflenet":
            self.model = models.shufflenet_v2_x0_5()
            self.model.fc = nn.Linear(1024,9)

        if model_path != None:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(torch.device('cpu'))
        self.model.eval()

    def inference(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        
        img = transform(img)
        
        prev = time.time()
        
        pred = self.model(img.unsqueeze(0))
        #pred = (SOFTMAX(pred) ** 0.5) * class_weight
        
        pred =SOFTMAX(pred)
        
        pred = torch.argmax(pred)
        

        now = time.time()

        return CLASSES[int(pred.detach())], (now - prev)

