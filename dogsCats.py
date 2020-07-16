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

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.conv3_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(2430, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        #F.log_softmax(x, dim=0)

categories = []
filename = os.listdir("folder sa podacima za treniranje")
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
                                      transforms.Resize((92,92)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train, valid_data = train_test_split(df,test_size=0.2)
train_data = DCDataset(train,"folder sa podacima za treniranje",train_transform)
valid_data = DCDataset(valid_data,"folder sa podacima za treniranje",train_transform)

epochs = 20
classes = 2
batch = 25
learning_rate = 0.001

train_loader = DataLoader(dataset=train_data,batch_size=batch,shuffle=True,num_workers=0)
valid_loader = DataLoader(dataset=valid_data,batch_size=batch,shuffle=True,num_workers=0)

device = torch.device('cuda')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
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
    model.train()
    for data, target in train_loader:
       
        data = data.to(device)
        target = target.to(device)

       
        optimizer.zero_grad()
        
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)

    
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        correct_valid += (predicted == target).sum().item()

        loss = criterion(output, target)

        
        valid_loss += loss.item() * data.size(0)

    
    train_loss = train_loss / len(train_loader.sampler)
    correct_train = correct_train / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    correct_valid = correct_valid / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    
    print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tCorrect training: {} \tCorrect valid: {}'.format(
        epoch, train_loss, valid_loss,correct_train,correct_valid))

model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

PATH = "gdje će se spremiti model"
torch.save(model.state_dict(),PATH)
