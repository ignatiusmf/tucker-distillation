from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100

import torch
import torchvision
import torchvision.transforms as transforms

import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt

from rich import print as pprint
from tqdm import trange

device = "cuda"
###############################################################################




fmap = torch.load("toolbox/problem_fm.pt", map_location=device, weights_only=True)
print(fmap.shape)


tl.set_backend("pytorch")
def tucker(feature_map): #expects 4d
    batch_size, channels, height, width = feature_map.shape
    core, factors = tl.decomposition.tucker(feature_map, rank=[batch_size, 32, 8, 8])
    return core



trouble = tucker(fmap)

print(trouble.shape)