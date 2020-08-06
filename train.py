import os
import torch
import random
import argparse
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import datasets, transforms, models


def load_func(train_dir, valid_dir, test_dir):
    '''
    func: to transform imaegs and load dataloaders from the directory
    return: trainloader,validateloader,testloader
    '''

    # Defining your transforms for the training, validation, and testing sets
    global train_image_datasets
    global test_image_datasets
    global valid_image_datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

    # loading the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir,transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)

    print("DataSet loading: Done");

    # Returning trainloader, validateloader, testloader
    return trainloader,testloader,validateloader


def setup_model(arch,lr_temp=0.001, hid_units=5000,gpu='gpu'):
    '''
    func: download the specified model and initialize the hidden layers, loss and optimizer
    return: model, criterion, optimizer
    '''
    # Build and train your network

    # Setting up the device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Selecting and importing the Model
    if(arch == 'vgg19'):
        model = models.vgg19(pretrained=True)
    elif(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print("invalid model")

    # Freezing the features
    for param in model.parameters():
        param.requires_grad = False

    # Creating our own classifier
    classifier = nn.Sequential(nn.Linear(25088,hid_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hid_units,1000),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(1000,102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    # Defining loss
    criterion = nn.NLLLoss()


    optimizer = optim.Adam(model.classifier.parameters(),lr=lr_temp)

    # Moving the model to the device available(i.e GPU or CPU)
    if (gpu == 'gpu'):
        model.to(device);
    print("Model Setup: Done");

    return model, criterion, classifier, optimizer


def train_func(model, optimizer, criterion, epochs, trainloader, validateloader, gpu):
    """
    Trains the given model
    """
    model.train(True)
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for images,labels in trainloader:
            steps+=1
            if (gpu == 'gpu'):
                images,labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if (steps % print_every == 0):
                model.eval()
                test_loss = 0
                accuracy = 0

                for images,labels in validateloader:

                    if (gpu == 'gpu'):
                        images,labels = images.to(device), labels.to(device)

                    logps = model(images)
                    loss = criterion(logps,labels)
                    test_loss+=loss.item()


                    # Accuracy part
                    # Referenced from the lecture material provided in AIPND
                    ps = torch.exp(logps)
                    top_ps,top_class = ps.topk(1,dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy+=torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validate loss: {test_loss/len(validateloader):.3f}.. "
                      f"Validate accuracy: {accuracy/len(validateloader):.3f}")
                running_loss = 0
                model.train()

    # Return model
    print("Model Training: Done");



def save_func(arch, epochs, optimizer, lr, model,classifier, checkpoint_path):
    """
    To save the trained model for further use
    """

    # Saving the checkpoint
    model.class_to_idx = train_image_datasets.class_to_idx
    checkpoint={'arch': arch,
                'input_size': 25088,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'lr': lr,
                'state_dict':model.state_dict(),
                'classifier': classifier,
                'class_to_idx': model.class_to_idx }
    torch.save(checkpoint, checkpoint_path)

    print("Model Saving: Done");



def start_fun():
    """
    To start the program
    """

    ag = argparse.ArgumentParser()
    ag.add_argument('data_dir',nargs='*', action='store',default="flowers")
    ag.add_argument('--arch', action='store', dest='arch', default= 'vgg19')
    ag.add_argument('--gpu',action='store',dest='gpu',default='gpu')
    ag.add_argument('--learning_rate',action='store',dest='learning_rate',default=0.001)
    ag.add_argument('--save_dir',action='store',dest='save_dir',default='checkpoint.pth')
    ag.add_argument('--epochs',action='store',dest='epochs',default=10)
    ag.add_argument('--hidden_units',action='store',dest='hid_units',default=5000)


    xyz = ag.parse_args()
    data_dir = xyz.data_dir
    arch = xyz.arch
    gpu = xyz.gpu
    lr = float(xyz.learning_rate)
    checkpoint_path = xyz.save_dir
    epochs = int(xyz.epochs)
    hid_units = int(xyz.hid_units)

    # Making dataset dir
    data_dir = data_dir[0]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # loading data loaders
    trainloader,testloader,validateloader = load_func(train_dir, valid_dir, test_dir)

    # Model Setup
    model, criterion,classifier, optimizer = setup_model(arch,lr,hid_units,gpu)

    # train Function
    train_func(model, optimizer, criterion, epochs, trainloader,validateloader, gpu)

    # Save model
    save_func(arch, epochs, optimizer, lr, model,classifier, checkpoint_path)
    print("train.py==finished: True")

start_fun();
