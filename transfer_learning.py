from torch.utils.data import Dataset
import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import time
import json
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 100
BATCH_SIZE = 512

model = torchvision.models.resnet152(pretrained=True)
import pdb; pdb.set_trace()
# model = model.eval().to(DEVICE)
# model = nn.DataParallel(model)
#
