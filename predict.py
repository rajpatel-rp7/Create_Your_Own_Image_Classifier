import numpy as np

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
    parser.add_argument('--checkpoint_dir', type = str, default = 'image_checkpoint.pth', 
                         help = 'path to save checkpoints')
    parser.add_argument('--device', type = str, default = 'gpu', 
                         help = 'device used')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                         help = 'json file with category names')
    parser.add_argument('--path_to_image', type = str, default = '99/image_07833.jpg', 
                         help = 'image to test')
    parser.add_argument('--top_k', type = int, default = 5, 
                         help = 'Top K classes')
    parser.add_argument('--arch', type = str, default = 'densenet121', 
                         help = 'Architecture used in training')
  
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch_name = checkpoint['arch']
    
    if (arch_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif (arch_name == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print("Architecture should be densenet121 or vgg16")
        
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer_dict']
    epochs = checkpoint['epoch']
        
    for param in model.parameters():
        param.requires_grad = False   
        
    return model, checkpoint['class_to_idx']

def process_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(path_to_image)
   
    pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_t = pre_process(pil_img)
    
    return img_t

def predict(path_to_image, model, device, top_k, class_to_idx):
     # TODO: Implement the code to predict the class from an image file
        
        
    if (device == 'gpu'):
        device = 'cuda'
    else:
        device = 'cpu'
        
    model.to(device)
    model.eval()
    img_t = process_image(path_to_image)
    # print(type(img_t), img_t.shape)
        
    img_t = img_t.unsqueeze_(0)
        
    img_t = img_t.to(device)
    
    with torch.no_grad():
        output = model.forward(img_t)
        ps2 = torch.exp(output)        
        top_p, top_class = ps2.topk(top_k)
        # class_to_idx = model.class_to_idx
        
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    # print ("IDX_to_Class:", idx_to_class)
    top_class = top_class.cpu().numpy()
    top_class1 = [idx_to_class[x] for x in top_class[0]]
    
    return top_p[0], top_class1


def main():
    
    # get all arguments
    args_values = get_arguments()
    data_dir = args_values.data_dir
    save_dir = args_values.checkpoint_dir
    device = args_values.device
    category_names = args_values.category_names
    path_to_image = args_values.path_to_image
    top_k = args_values.top_k
    arch = args_values.arch
    
    # save directory
    image_directory_path = 'flowers/test/' + path_to_image
    
    print("Data Directory:", data_dir)
    print("Save Directory:", save_dir)
    print("Device:", device)
    print("category_names:", category_names)
    print("path_to_image:", image_directory_path)
    print("top_k:", top_k)
    
   
    
    # if checkpoint exists complete prediction
    if os.path.exists(save_dir):
        print("Checkpoint exists")
        
        # Category to name
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        # load saved checkpoint
        loaded_model, class_to_idx = load_checkpoint(save_dir)
        print("Model loaded")
        # Call predict function 
        probs, classes = predict(image_directory_path, loaded_model, device, top_k, class_to_idx)
        
        img_number = image_directory_path.split('/')[-2]
        # print("Img Number and type ", img_number, ";", type(img_number))
        
        actual_image_name = cat_to_name[str(img_number)]
        print("Actual Image Name:", actual_image_name)
        # print("Predict Completed*****")
        print("Probs:", probs)
        print("classes:", classes)
                    
        arc1 = np.array(classes)
        labels = [cat_to_name[str(x)] for x in arc1]
        print("Labels:", labels)
        print("Prediction completed")      
    else:
        print("Could not find checkpoint file")
    


if __name__ == '__main__':
    main()