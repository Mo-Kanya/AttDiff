import argparse
import json
import os
import time
import sys
import copy
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# Dataset utils
sys.path.append('..')
from data.Kaggle_FFHQ_Resized_256px import ffhq_utils
from data.CUB.cub2011 import Cub2011

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
trainform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
valform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_dataset = Cub2011('/root/Explaining-In-Style-Reproducibility-Study/data/CUB', train=True, transform=trainform)
valid_dataset = Cub2011('/root/Explaining-In-Style-Reproducibility-Study/data/CUB', train=False, transform=valform)
dataset_sizes = {
    "train":len(train_dataset),
    "val": len(valid_dataset)
}

train_loader = DataLoader(train_dataset, batch_size=256, pin_memory=True, num_workers=6, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=256, pin_memory=True, num_workers=6)


model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)
model.classifier[1] = nn.Linear(1280, 200).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
criterion = criterion.to(device)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}

dataloaders = {
    "train":train_loader,
    "val": val_loader
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                training_history['accuracy'].append(epoch_acc)
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(epoch_acc)
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model, criterion, optimizer, step_scheduler, num_epochs=25)
torch.save(model_ft.state_dict(), os.path.join("saved_models", "mobilenet-CUB200.pth"))
