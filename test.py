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


Data = Cifar100()
trainloader, testloader = Data.trainloader, Data.testloader

model_path = r"toolbox/Cifar100_ResNet112.pth"
model = ResNet56(100).to(device)
# model.load_state_dict(torch.load(model_path, weights_only=True)["weights"])
model.eval()
print("Loaded model")



tl.set_backend("pytorch")
def tucker(feature_map): #expects 4d
    batch_size, channels, height, width = feature_map.shape
    core, factors = tl.decomposition.tucker(feature_map, rank=[batch_size, 32, 8, 8])
    return core

for i in range(100):
    print(i)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        current_output = outputs[2] 

        tucker_fmap = tucker(current_output)

        if batch_idx % 10 == 0:
            print(batch_idx)
            print(tucker_fmap.shape)