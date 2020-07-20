import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.image as img
from torchvision import models

class DCDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

categories = []
filename = os.listdir("")
for file in filename:
    category = file.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename' : filename,
    'category' : categories
})


train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(256),
                                      transforms.RandomCrop(254),
                                      transforms.ColorJitter(),
                                      transforms.Resize((92,92)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4883, 0.4551, 0.4170), (0.2208, 0.2161, 0.2163))])


train, valid_data = train_test_split(df,test_size=0.2)
train_data = DCDataset(train,"",train_transform)
valid_data = DCDataset(valid_data,"",train_transform)

epochs = 2
classes = 2
batch = 20
learning_rate = 0.001

train_loader = DataLoader(dataset=train_data,batch_size=batch,shuffle=True,num_workers=0)
valid_loader = DataLoader(dataset=valid_data,batch_size=batch,shuffle=True,num_workers=0)

device = torch.device('cpu')
model_ft = models.vgg16(pretrained=True)
model_ft.classifier[6] = nn.Linear(4096,classes)
feature_extract = True


model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params_to_update,lr = learning_rate,momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

train_losses = []
valid_losses = []


for epoch in range(1, epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    correct_valid = 0

    # training-the-model
    model_ft.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model_ft(data)
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model_ft.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model_ft(data)
        _, predicted = torch.max(output.data, 1)
        correct_valid += (predicted == target).sum().item()

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    correct_train = correct_train / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    correct_valid = correct_valid / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tCorrect training: {} \tCorrect valid: {}'.format(
        epoch, train_loss, valid_loss,correct_train,correct_valid))

model_ft.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

PATH = ""
torch.save(model_ft.state_dict(),PATH)
