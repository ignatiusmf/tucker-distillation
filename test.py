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
EPOCHS=150
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
    return core, factors

def compute_core(feature_map, factors):
    return tl.tenalg.multi_mode_dot(feature_map, [f.T for f in factors], modes=[0, 1, 2, 3])

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

BETA = 125

def evaluate_model(model, loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[3], targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    print(f'TEST | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    return val_loss/(b_idx+1), correct*100/total


import matplotlib.pyplot as plt
import numpy as np
def plot_the_things(train_loss, test_loss, train_acc, test_acc):
        plt.plot(np.log10(np.array(train_loss)), linestyle='dotted',color='b', label=f'Train Loss')
        plt.plot(np.log10(np.array(test_loss)), linestyle='solid',color='b', label=f'Test Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Log10 Loss')
        plt.legend()
        plt.savefig(f'logs/Loss.png')
        plt.close()

        max_acc = np.max(np.array(test_acc))

        plt.plot(np.array(train_acc), linestyle='dotted',color='r', label=f'Train Accuracy')
        plt.plot(np.array(test_acc), linestyle='solid',color='r', label=f'Test Accuracy')

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 105, 5))
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.axhline(y=max_acc, color='black', linestyle='-', linewidth=0.5)
        plt.text(0, max_acc + 1, f"Max Acc = {max_acc}", color='black', fontsize=8)


        plt.legend()
        plt.savefig(f'logs/Accuracy.png')
        plt.close()

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
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        teacher_tucker, teacher_factors = tucker(teacher_outputs[2])
        # student_tucker, student_factors = tucker(student_outputs[2])
        student_tucker = compute_core(student_outputs[2], teacher_factors)

        tucker_loss = BETA * F.l1_loss(FT(teacher_tucker), FT(student_tucker))
        hard_loss = F.cross_entropy(student_outputs[3], targets)
        loss = tucker_loss + hard_loss

        print(loss)

        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    scheduler.step()

    print(f'TRAIN | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(correct*100/total)

    tel, test_accuracy = evaluate_model(student, testloader)
    test_loss.append(tel)
    test_acc.append(test_accuracy)

    if test_accuracy > max_acc:
        max_acc = test_accuracy
        torch.save({'weights': student.state_dict()}, f'logs/ResNet56.pth')
    
    plot_the_things(train_loss, test_loss, train_acc, test_acc)