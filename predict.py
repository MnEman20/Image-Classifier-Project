# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict
import time
import copy
from torch.optim import lr_scheduler
from PIL import Image
import PIL
import numpy as np
import argparse
import json
from train import load_checkpoint as LC
import math
#import train

parser = argparse.ArgumentParser(description='Load a trained network and predict the type of flower from a jpeg')
parser.add_argument('-g', '--gpu', action="store", help='train on GPU')
parser.add_argument('-lc', '--load_checkpoint', action="store", default="/home/workspace/ImageClassifier/checkpoint_mv.pth")
parser.add_argument('-ipf', '--image_path_flw', action="store") #'/home/workspace/ImageClassifier/flowers/test/63/image_05878.jpg
parser.add_argument('-json', '--json_file', action="store") #'/home/workspace/ImageClassifier/cat_to_name.json'
args = parser.parse_args()

gpu = args.gpu
checkpoint_path = args.load_checkpoint
image_path_flw = args.image_path_flw
json_path = args.json_file
device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")

with open(json_path, 'r') as f:
    cat_to_name = json.load(f)

os.chdir('/home/workspace/ImageClassifier')
model = LC('/home/workspace/ImageClassifier/checkpoint_mv.pth')

#image_path_flw = '/home/workspace/ImageClassifier/flowers/test/63/image_05878.jpg'

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # new required method - 2/26/2019

    width, height = image_path.size
    #print(width, height)
    #print(image_path.size)
    size = 256, 256
    if width > height:
        ratio = float(width) / float(height)
        #print(ratio)
        newheight = ratio * size[0]
        #print(newheight)
        image = image_path.resize((size[0], int(math.floor(newheight))), Image.ANTIALIAS)
        #print(image)

    else:
        ratio = float(height) / float(width)
        #print(ratio)
        newheight = ratio * size[1]
        #print(newheight)
        image = image_path.resize((size[1], int(math.floor(newheight))), Image.ANTIALIAS)
        #print(image)



    left = (image.width - 224)/2
    top = (image.height - 224)/2
    right = (image.width + 224)/2
    bottom = (image.height + 224)/2
    image = image.crop((left, top, right, bottom))
    #print(image.size)


    #image_tensor = torch.from_numpy(image).type(torch.FloatTensor)

    #img = np.array(image_path)/255
    img = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    img = img.transpose((2, 0, 1))



    # prior method - 2/26/2019
    """adjust_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image_tensor = adjust_image(image_test)"""


    #print(img)

    return img


"""
# pre-defined Udacity funtion imshow
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
"""
image_test = Image.open(image_path_flw)
image = process_image(image_test)
#imshow(image)

# TODO: Implement the code to predict the class from an image file

def predict(image_path, model, topk, device):
    if gpu == "gpu":
        model.to(device)
        img_torch = process_image(image_path)
        img_torch = torch.from_numpy(img_torch).type(torch.cuda.FloatTensor)
        img_torch.unsqueeze_(0)
        prob = model.forward(img_torch.cuda())
        prob = torch.exp(prob)
        
    else:
        model.to(device)
        img_torch = process_image(image_path)
        img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
        img_torch.unsqueeze_(0)
        #print(img_torch.shape)
        #img_torch.transpose
        prob = model.forward(img_torch)
        #print(prob)
        prob = torch.exp(prob)
   
    # kept getting unhashable type: 'numpy.ndarray' error so put this code in to detach array
  
    if gpu == "gpu":
        prob = prob.cpu()
        probability = prob.detach().numpy().tolist()[0]
    else:
        probability = prob.detach().numpy().tolist()[0]
    probability, label = prob.topk(topk)
    label = label.detach().numpy().tolist()[0]


    idx_to_classes = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_classes[lab] for lab in label]
    flowers = [cat_to_name[idx_to_classes[lab]] for lab in label]
    #print(idx_to_classes)
    #print(classes)
    """


    #print(img_tensor)

    probability = torch.exp(model.forward(img_tensor))

    ## prior method 2/26/2019
    #with torch.no_grad():
       # output = model.forward(img_torch.cuda())
        #print(output)

    #probability = F.softmax(output.data,dim=1)

    #print(probability)
    
    # method from review 2/26/2019 - not sure what this does, was not explained by instructor

    topk_class = probability.cpu().numpy()[0]
    idx_to_classes = {v: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_classes[idx] for idx in topk_class]

    return classes.topk(topk)
    """
    return probability, classes, flowers


predict(image_test, model,5, device)


# TODO: Display an image along with the top 5 classes
def plot_check():
    #plt.figure(figsize = (5, 5))

    probabilities, labels, flowers = predict(image_test, model, 5, device)
    image = process_image(image_test)

    #axs = imshow(image, ax = plt,)
    #axs.axis('on')

    a = probabilities
    a = a.detach().numpy().tolist()[0]
    b = flowers


    print("Flower Name & Probability:\n ",b[0], " - ", a[0],"\n",b[1], " - ", a[1],"\n",b[2], " - ", a[2])

    """
    axs.title(b[0])
    axs.show()

    N=float(len(b))
    fig,ax = plt.subplots(figsize=(8,5))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(b)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.25,0.5,0.75,1])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    fig.savefig('output.png')
    #plt.show()
    """

plot_check()
