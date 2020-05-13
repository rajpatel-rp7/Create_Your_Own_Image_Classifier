import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import json

import argparse
import os
import time

from collections import OrderedDict


def get_arguments():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                         help = 'path to the test, train and valid data')
    parser.add_argument('--learn_rate', type = str, default = '0.001', 
                         help = 'Learning rate for the model')
    parser.add_argument('--epochs', type = int, default = 4, 
                         help = 'epochs required for model')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint', 
                         help = 'path to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'densenet121', 
                         help = 'model architecture')
    parser.add_argument('--device', type = str, default = 'gpu', 
                         help = 'device used')
  
    return parser.parse_args()

def get_transforms(train_dir, valid_dir, test_dir): 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    img_datasets = dict()
    img_datasets['train_data'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    img_datasets['test_data'] = datasets.ImageFolder(test_dir, transform=test_transforms)
    img_datasets['validation_data'] = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders 
    dataloaders = dict()
    dataloaders['trainloader'] = torch.utils.data.DataLoader(img_datasets['train_data'], batch_size=64, shuffle=True)
    dataloaders['testloader'] = torch.utils.data.DataLoader(img_datasets['test_data'], batch_size=32, shuffle=True)
    dataloaders['validloader'] = torch.utils.data.DataLoader(img_datasets['validation_data'], batch_size=20, shuffle=True)
    
    return img_datasets, dataloaders

def train_save_model(args, device):
    
    arch = args.arch
    data_dir = args.data_dir
    epochs = args.epochs
    save_dir = args.save_dir
    arch = args.arch
    
    # print("Device in train save model:", device)
    
    # get data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # print("Training network started.......")
    start_time = time.time()

    # model architecture
    if (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif (arch == 'vgg16' ): 
        model = models.vgg16(pretrained=True)
    else:
        print("Only densenet121 and vgg16 supported. Please try again.")
    
    for param in model.parameters():
        param.requires_grad = False    
  
    # Get features
    if (arch == 'vgg16'):
        num_features = model.classifier[0].in_features
    elif (arch == 'densenet121'):
        num_features = model.classifier.in_features        
    else:
        print("Model not supported")
        
    # add nother classifier if this one does not work
    classifier = nn.Sequential(OrderedDict([                           
                          ('fc1', nn.Linear(num_features, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 102)), # 102 output categories
                          ('output', nn.LogSoftmax(dim = 1))
                          ]))
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device);
    print("Model and optimizer completed and training started.")
    
    # Start Classifier training    
    img_datasets, dataloaders = get_transforms(train_dir, valid_dir, test_dir)
            
    steps = 0
    running_loss = 0
    print_every = 20
    print("Started trainer.......")
    for epoch in range(epochs):
        for inputs, labels in dataloaders['trainloader']:
            steps += 1
            # print("Step:", steps)
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            # print("Inputs Shape:", inputs.shape)
            # print("Labels Shape:", labels.shape)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
        
            # print("logps shape:", logps.shape)
        
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                #set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    for inputs2, labels2 in dataloaders['validloader']:
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        logps = model.forward(inputs2)
                        batch_loss = criterion(logps, labels2)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(dataloaders['validloader']):.3f}.. "
                    f"Validation accuracy: {accuracy/len(dataloaders['validloader']):.3f}")
                running_loss = 0
                model.train()           
                
    end_time = time.time()
    total_time =  end_time - start_time
    print("Total testing time:", int(total_time/60), "min ",  ":", int(total_time%60), "sec")
    return model, optimizer, classifier            
 
def testing(model, testloader, device):
    test_loss = 0
    accuracy = 0
    steps_test = 0
        
    start_time = time.time()
    
    with torch.no_grad():
        model.to(device)
        model.eval()
        for inputs3, labels3 in testloader:
            steps_test +=1
            inputs3, labels3 = inputs3.to(device), labels3.to(device)
            logps = model(inputs3)
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels3.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
    end_time = time.time()
    total_time = end_time - start_time
    print("Total testing time:", int(total_time/60), "min  :", int(total_time%60), "sec")
    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def save_checkpoint(train_data, epochs, trainloader, model, optimizer, classifier, arch):
    model.class_to_idx = train_data.class_to_idx
    model.epochs = epochs
    checkpoint = {'input_size': [3, 224, 224],
                'batch_size': trainloader.batch_size,
                'output_size': 102,
                'state_dict': model.state_dict(),
                'optimizer_dict':optimizer.state_dict(),
                'class_to_idx': model.class_to_idx,
                'epoch': model.epochs,
                'classifier': classifier, 
                'arch': arch}
    torch.save(checkpoint, 'image_checkpoint.pth')
    return checkpoint
    
def main():
    # Get arguments and print
    args_values = get_arguments()
    data_dir = args_values.data_dir
    learn_rate = args_values.learn_rate
    epochs = args_values.epochs
    save_dir = args_values.save_dir
    arch = args_values.arch
    device = args_values.device
    
    print("Data Directory1:", data_dir)
    print("learn_rate:", learn_rate)
    print("epochs:", epochs)
    print("save_dir:", save_dir)
    print("arch:", arch)
    print("device:", device)
    
    # set data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transforms
    img_datasets, dataloaders = get_transforms(train_dir, valid_dir, test_dir)
    print("Datasets transforms completed")
    
    # Cat_to_name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    print("Cat_to_name completed")
    
    # Device
    # CPU or gpu
    if (device == 'gpu'):
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cuda'
    else:
        device = 'cpu'
    
      
    model, optimizer, classifier = train_save_model(args_values, device)
    
    print("Training Completed and testing started.")
    
    testing(model, dataloaders['testloader'], device)
    
    print("*****Testing Completed and saving checkpoint***********")
    
    checkpoint = save_checkpoint(img_datasets['train_data'], epochs, dataloaders['trainloader'], model, optimizer, classifier, arch)
    
    print ("************Checkpoint completed and saved******************")
    
        
if __name__ == '__main__':
    main()

