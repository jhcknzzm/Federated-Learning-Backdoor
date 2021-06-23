import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from EMNIST_model import *
# Training settings
import numpy as np
from PIL import Image

class ArdisDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None, train = True):

        if train:
            X = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_2828.csv', dtype='float')
            Y = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_labels.csv', dtype='float')
        else:
            X = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_test_2828.csv', dtype='float')
            Y = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_test_labels.csv', dtype='float')
        Y = np.argmax(Y,axis = 1)

        X = X[Y==7]

        self.X  = X

        self.transform = transform
        self.attack_target = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        img = self.X[index]
        img = np.reshape(img, (28,28))
        print(img.shape)
        # img = img.cpu().numpy()
        img = Image.fromarray(img)

        target = int(self.attack_target)

        if self.transform is not None:
            img = self.transform(img)


        return img, target

batch_size = 64

transform_labeled = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=28,
                          padding=int(28*0.125),
                          padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = ArdisDataset( transform=transform_labeled, train=True)

test_dataset = ArdisDataset( transform=transform_val, train=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):


        output = model(data)
        #output:64*10

        loss = F.nll_loss(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
train(1)
