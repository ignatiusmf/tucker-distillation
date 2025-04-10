from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt

from rich import print as pprint
from tqdm import trange

device = "cuda"
EPOCHS=50
###############################################################################

# TEACHER
model_path = r"toolbox/Cifar100_ResNet112.pth"
teacher = ResNet112(100).to(device)
teacher.load_state_dict(torch.load(model_path, weights_only=True)["weights"])
teacher.eval()

# STUDENT 
student = ResNet56(100).to(device)
student.train()


optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


Data = Cifar100()
trainloader, testloader = Data.trainloader, Data.testloader

tl.set_backend("pytorch")
def tucker(feature_map): #expects 4d
    batch_size, channels, height, width = feature_map.shape
    core, factors = tl.decomposition.tucker(feature_map, rank=[batch_size, 32, 8, 8])
    return core

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

BETA = 125

for i in range(EPOCHS):
    print(i)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        teacher_tucker = tucker(teacher_outputs[2])
        student_tucker = tucker(student_outputs[2])

        tucker_loss = BETA * F.l1_loss(FT(teacher_tucker), FT(student_tucker))
        hard_loss = F.cross_entropy(student_outputs[3], targets)
        loss = tucker_loss + hard_loss

        loss.backward()
        optimizer.step()

    scheduler.step()