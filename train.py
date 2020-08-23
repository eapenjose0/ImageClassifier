#imports 

import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from workspace_utils import active_session


def arg_parser():
    '''
    Parses the input arguements
       
    Output:
        - args: parsed arguements
    '''
    parser = argparse.ArgumentParser(description="Configs")
        
    parser.add_argument('--arch', 
                        type=str, 
                        help='Define the base nerual network architecture')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Learning Rate')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Number of hidden units')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training')
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU instead of CPU')
    args = parser.parse_args()
    return args


def getTrainLoader(train_dir):
    '''
        Apply Transformation to the train dataset and returns a trainLoader
    '''
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=60, shuffle=True)
    return trainloader, train_datasets


def getTestLoader(test_dir):
    '''
        Apply Transformation to the test/validation dataset and returns a testLoader
    '''
    test_transforms =  transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)  
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=60)
    return testloader
    

def getDevice(gpu_arg):
    '''
        Return the device for performing the training.
    '''
    
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("GPU not available! Swithing to CPU")
    return device


def getBaseModel(arch):
    '''
        Return the base model for performing the training. 
        By default vgg16 will be used as the base model
    '''
    
    if type(arch) == type(None):
        arch = "vgg16"
        model = models.vgg16(pretrained=True)
        model.name = arch
    else: 
        exec("model = models.{}(pretrained=True)".format(arch))
        model.name = arch

    for param in model.parameters():
        param.requires_grad = False
    
    return model


def setModelClassifier(model, hidden_units):
    '''
        Sets the classifier for traing the model
    '''
    
    if type(hidden_units) == type(None):
        hidden_units = 4096
    else: 
        hidden_units = hidden_units
        
        
    inp_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inp_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drp1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
 
    # Replace the classifier in pretrained model with a new classifier
    model.classifier = classifier
    
    return model


def trainModel(model, trainloader, validloader, device, epochs, learning_rate):
    '''
        Trains the model based on the specified hyperparameter
    '''
    
    if type(learning_rate) == type(None):
        lr = 0.001
    else: 
        lr = learning_rate
        
    if type(epochs) == type(None):
        epochs = 5
    else: 
        epochs = epochs    
    
    steps = 0
    running_loss = 0
    print_every = 30
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    return model


def testModel(model, testloader, device):
    '''
        Tests the model and returns the accuracy of the final model.
    '''
    
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
           
    print(f"Test accuracy: {accuracy/len(testloader):.3f}")    

    
def saveModel(model, train_datasets):
    '''
        Saves the model as checkpoint in the current directory
    '''
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'base_model': model.name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'model_checkpoint.pth')


def main():
     

    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Get dataloaders
    trainloader, train_datasets = getTrainLoader(train_dir)
    validloader = getTestLoader(valid_dir)
    testloader = getTestLoader(test_dir)
    
    
    # Get base model
    model = getBaseModel(arch=args.arch)
    
    # Set model classifier
    model = setModelClassifier(model, hidden_units=args.hidden_units)
     
    # Set device
    device = getDevice(gpu_arg=args.gpu);
    print("Device: ", device)
    
    # Send model to device
    model.to(device);
    
    # Train the model
    print("Model training started!")
    model = trainModel(model, trainloader, validloader, device, args.epochs, args.learning_rate)
    print("Model training completed!")
    
    # Test the model
    testModel(model, testloader, device)
    
    # Save the model
    saveModel(model, train_datasets)


if __name__ == '__main__':
    main()