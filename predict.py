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
#import train

parser = argparse.ArgumentParser(description='Load a trained network and predict the type of flower from a jpeg')
parser.add_argument('-g', '--gpu', action="store", default="gpu", help='train on GPU')
parser.add_argument('-lc', '--load_checkpoint', action="store", default="/home/workspace/ImageClassifier/checkpoint_mv.pth")
parser.add_argument('-ipf', '--image_path_flw', action="store")
args = parser.parse_args()

gpu = args.gpu
checkpoint_path = args.load_checkpoint
image_path_flw = args.image_path_flw
device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")

with open('/home/workspace/ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
      
os.chdir('/home/workspace/ImageClassifier')
model = LC('/home/workspace/ImageClassifier/checkpoint_mv.pth')

#image_path_flw = '/home/workspace/ImageClassifier/flowers/test/63/image_05878.jpg'

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image_test = Image.open(image_path)
    
    adjust_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = adjust_image(image_test)
    
    return image_tensor
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
#image_path = 'test/100/image_07896.jpg'
image = process_image(image_path_flw)
#imshow(image)

# TODO: Implement the code to predict the class from an image file

def predict(image_path, model, topk=5):   
    model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        #print(output)
        
    probability = F.softmax(output.data,dim=1)
    
    #print(probability)
    
    
    return probability.topk(topk)

predict(image_path_flw, model, 5)


# TODO: Display an image along with the top 5 classes
def plot_check():
    #plt.figure(figsize = (5, 5))
         
    probabilities = predict(image_path_flw, model, 5)
    image = process_image(image_path_flw)
    
    #axs = imshow(image, ax = plt,)
    #axs.axis('on')
    #for i in x:
        #print(type(i.data[0]))
        #print(i.size())
        #print(i)
    #print(i)
    
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(i+1)] for i in np.array(probabilities[1][0])]
    
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

