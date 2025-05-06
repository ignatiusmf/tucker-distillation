from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100
from toolbox.utils import plot_the_things, evaluate_model

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from pathlib import Path
import argparse

DEVICE = "cuda"

# Hyperparameters
EPOCHS = 150
BETA = 125
BATCH_SIZE = 128*4

class GenerationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

gen_module = GenerationModule(in_channels=64).to(DEVICE)

def feature_map_distillation(teacher_outputs, student_outputs, targets):
    beta = 125
    teacher_fmap = teacher_outputs[2]
    student_fmap = student_outputs[2]

    noise = torch.normal(mean=0.0, std=1.0, size=teacher_fmap.shape, device=DEVICE)
    noisy_student_fmap = student_fmap + noise
    generated_student_fmap = gen_module(noisy_student_fmap)

    brute_loss = 125 * F.l1_loss(FT(teacher_fmap), FT(generated_student_fmap))
    hard_loss = F.cross_entropy(student_outputs[3], targets)
    return brute_loss + hard_loss

experiment_path = 'noise'
Path(f"experiments/{experiment_path}").mkdir(parents=True, exist_ok=True)


# Model setup
model_path = r"toolbox/Cifar100_ResNet112.pth"
teacher = ResNet112(100).to(DEVICE)
teacher.load_state_dict(torch.load(model_path, weights_only=True)["weights"])

student = ResNet56(100).to(DEVICE)

Data = Cifar100(BATCH_SIZE)
trainloader, testloader = Data.trainloader, Data.testloader

optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

for i in range(EPOCHS):
    print(i)
    teacher.eval()
    student.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()


        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)
        loss = feature_map_distillation(teacher_outputs, student_outputs, targets)
        print(loss)

        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    scheduler.step()

    trl, tra = val_loss/(b_idx+1), 100*correct/total
    print(f'TRAIN | Loss: {trl:.3f} | Acc: {tra:.3f} |')
    tel, tea = evaluate_model(student, testloader)

    train_loss.append(trl)
    train_acc.append(tra)
    test_loss.append(tel)
    test_acc.append(tea)

    if tea > max_acc:
        max_acc = tea
        torch.save({'weights': student.state_dict()}, f'experiments/{experiment_path}/ResNet56.pth')
    
    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_path)