import os
import json
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

def load_checkpoint(filepath):
    """
    Loads the saved model checkpoints
    """
    checkpoint = torch.load(filepath)

    # https://www.programiz.com/python-programming/methods/built-in/getattr
    # Used above link to access the model below
    model = getattr(models,checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    lr = checkpoint['lr']

    return model

def process_image(image):
    """
    func: Processes the input image
    """

    img = Image.open(image)

    img = img.resize((256,256))
    (width,length) = 256,256

    left = (width - 224) / 2
    top = (length - 224) / 2
    right = width - (width - 224) / 2
    bottom = length - (length - 224) / 2
    img = img.crop((left, top, right, bottom))

    img = np.array(img)
    img = img / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    return img


def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()

    #https://pytorch.org/docs/stable/autograd.html
    #used above link(having syntax and uses of Variable) to perform the necessary tasks. Variable used below
    image = Variable(image)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.float()
    with torch.no_grad():
        c = model(image)
        ps = torch.exp(c)

    top_five_ps,top_five_class = ps.topk(topk, dim=1)
    top_five_ps = top_five_ps.tolist()[0]
    top_five_class = top_five_class.tolist()[0]

    prob = torch.topk(ps, topk)[0].tolist()[0] # probabilities
    index = torch.topk(ps, topk)[1].tolist()[0]


    xyz = []
    l = len(model.class_to_idx.items())
    for i in range(l):
        xyz.append(list(model.class_to_idx.items())[i][0])
    label = []
    for i in range(topk):
        label.append(xyz[top_five_class[i]])
    return top_five_ps,label

def start_func():
    img = random.choice(os.listdir('./flowers/test/22/'))
    image_path = './flowers/test/22/' + img
    ag = argparse.ArgumentParser();
    ag.add_argument('input_image', nargs='*',action='store', default=image_path)
    ag.add_argument('checkpoint', nargs='*',action='store', default='checkpoint.pth')
    ag.add_argument('--top_k', dest='top_k',action='store', default=5)
    ag.add_argument('--gpu',action='store',dest='gpu',default='gpu')
    ag.add_argument('--category_names',action='store',dest='category_names',default='cat_to_name.json')

    xyz = ag.parse_args()
    input_image = xyz.input_image[0]
    checkpoint_path = xyz.checkpoint
    topk = xyz.top_k
    gpu = xyz.gpu
    category_names = xyz.category_names



    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if gpu =='gpu':
        model.to(device);
    topk = int(topk)
    ps_list,class_list = predict(input_image, model, topk)

    # Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    max_prob_index = np.argmax(ps_list)
    classes = class_list[max_prob_index]
    labels = []
    for i in class_list:
        labels.append(cat_to_name[i])
    for i in range(topk):
        print('{}:  {}'.format(labels[i], ps_list[i]))
start_func();
