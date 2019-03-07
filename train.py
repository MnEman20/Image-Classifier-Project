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

#command line inputs
parser = argparse.ArgumentParser(description='Train a Neural Network using VGG16 or Densenet121, input "gpu" to train on GPU; - example Black Eyed Susan /home/workspace/ImageClassifier/flowers/test/63/image_05878.jpg with 99% accuracy')
parser.add_argument('-a', '--architecture', action="store",default="VGG16", type=str, help='Training Architecture Type')
parser.add_argument('-l', '--learning_rate', action="store", default=0.001, help='Training Learning Rate, VGG16 works best with 0.001')
parser.add_argument('-h_1', '--hidden_layer1', action="store", default=4096, type=int, help='1st hidden layer value, VGG16 default 4096')
parser.add_argument('-h_2', '--hidden_layer2', action="store", default=1000, type=int, help='2nd hidden layer value, VGG16 default 1000')
parser.add_argument('-d', '--dropout', action="store", default=0.5, help='dropout rate of trainer, VGG16 default 0.5')
parser.add_argument('-e', '--epochs', action="store", default=10, help='number of epochs to train, default 10')
parser.add_argument('-g', '--gpu', action="store", help='train on GPU or CPU, input "gpu" else it will train on CPU(not advisable)')
parser.add_argument('-sc', '--save_checkpoint', action="store", default="/home/workspace/ImageClassifier/checkpoint_mv.pth")
parser.add_argument('-dir', '--data_dir', action="store", default="/home/workspace/ImageClassifier/flowers")
parser.add_argument('-ipf', '--image_path_flw', action="store")
parser.add_argument('-json', '--json_file', action="store") #'/home/workspace/ImageClassifier/cat_to_name.json'

args = parser.parse_args()
arch = args.architecture
lr = args.learning_rate
hidden_layer1 = args.hidden_layer1
hidden_layer2 = args.hidden_layer2
dropout = args.dropout
num_epochs = args.epochs
gpu = args.gpu
checkpoint_path = args.save_checkpoint
path = args.data_dir
image_path_flw = args.image_path_flw
json_path = args.json_file


arch_num = {"VGG16":25088,
            "Densenet121":1024}

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")
data_dir = path
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
os.chdir(data_dir)
cwd = os.getcwd()
#print(cwd)

# TODO: Define your transforms for the training, validation, and testing sets
# I used a transfer learning tutorial from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
## which taught me how to define sub-transformations within the main variable data_transforms (2/5/2019)
# I had to change the current working directory to get the data transforms pointed correctly (2/5/2019)
data_transforms = {
    train_dir: transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    valid_dir: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    test_dir: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
}
# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [train_dir, valid_dir, test_dir]}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in [train_dir, valid_dir, test_dir]}

dataset_sizes = {x: len(image_datasets[x])
                              for x in [train_dir, valid_dir, test_dir]}

class_names = image_datasets[train_dir].classes

with open('/home/workspace/ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
if arch == 'VGG16':
                model = models.vgg16(pretrained=True)
elif arch == 'Densenet121':
                model = models.densenet121(pretrained=True)
else:
                print("{} not a valid model. Use only VGG16 or Densenet121".format(arch))

#model

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('dropout', nn.Dropout(dropout)),
    ('fc1', nn.Linear(arch_num[arch], hidden_layer1)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
    ('relu', nn.ReLU()),
    ('fc3', nn.Linear(hidden_layer2, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# tutorial from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#training-the-model helped me define a
## function to perform the model training. (2/6/2019)
# model is from vgg16, criterion is NLLLoss, optimizer is Adam to update the weights, scheduler lr_scheduler to adjust
## the learning rate and step size, epochs are feed forward and backward through the network, cuda is GPU

#forward pass, backward pass, weight update
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train_dir, valid_dir]:
            if phase == train_dir:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train_dir):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == train_dir:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == valid_dir and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("Total training time in minutes: ", total_time)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#model = model.to('cuda')
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
lrschedule = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
#num_epochs=10

# Greatest accuracy found for the validation and test sets was optimizer learning rate of 0.001, lrschedule step size of 4
## with gamma 0.1 and epochs 10. Other test variables never yielded higher than 45% accuracy for validation set
if __name__ == '__main__':
    model_train = train_model(model, criterion, optimizer, lrschedule, num_epochs)

# TODO: Do validation on the test set
# used almost the same function as in training and validation but with torch.no_grad and model.forward(inputs).
def network_test(model, cuda=False):
    start_time = time.time()

    # test phase
    for phase in [test_dir]:
        phase == test_dir
        #model.eval()  # Set model to evaluate mode
        model.to(device='cuda')

        # forward
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[test_dir]):
                if cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model.forward(inputs)
                    _, predictions = torch.max(outputs, 1)
                    total_images = predictions == labels.data

        # statistics

                print('Test directory images Accuracy: %d %%' % (100*total_images.float().mean()))



    end_time = time.time()
    total_time = (end_time - start_time)
    print("Total testing time in seconds: ", total_time)

    # load best model weights
    return model

# accuracy results > 84%.
if __name__ == '__main__':
    test_network = network_test(model, True)

# TODO: Save the checkpoint
os.chdir('/home/workspace/ImageClassifier/')
if __name__ == '__main__':
    model.class_to_idx = image_datasets[train_dir].class_to_idx
    model.cpu()
    torch.save({'arch': arch,
                'hidden_layer1': hidden_layer1,
                'hidden_layer2': hidden_layer2,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx},
            'checkpoint_mv.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch'] == 'VGG16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer.state_dict = checkpoint['optimizer_state_dict']
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(arch_num[arch], hidden_layer1)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(hidden_layer2, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        return model

    elif checkpoint['arch'] == 'Densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer.state_dict = checkpoint['optimizer_state_dict']
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(arch_num[arch], hidden_layer1)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(hidden_layer2, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        print("wrong architecture input")



    #optimizer.state_dict = checkpoint['optimizer_state_dict']


#model = load_checkpoint('checkpoint_mv.pth')
