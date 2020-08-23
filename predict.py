import argparse
import json
import PIL
import torch
import numpy as np
from torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument('--image', 
                        type=str, 
                        help='Input image path',
                        required=True)

    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Model checkpoint path',
                        required=True)
    
    parser.add_argument('--top_k', 
                        type=int, 
                        help='top k probabilities')
    
    parser.add_argument('--category_names', 
                        type=str, 
                        help='map labels to real names',
                        required=True)

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='use gpu for prediction')

    args = parser.parse_args()
    
    return args


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


def load_model_from_checkpoint(file):
    """
    Loads the model from a checkpoint file.
    
    inputs:
        - file: input file name
        
    output:
        - model: model loaded from the checkpoint file
    """
    
    checkpoint = torch.load("model_checkpoint.pth")
    
    if checkpoint['base_model'] == 'vgg16':
        model = models.vgg16(pretrained=True);
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
        input: 
            - image_path: path to the input image
        output:
            - np_image: processed image converted to numpy array
    '''
    
    image = PIL.Image.open(image_path)
    width, height = image.size
    
    
    # Resizing
    aspect_ratio = width/height
    
    if width < height:
        new_width = 256
        new_height = new_width/aspect_ratio
    else:
        new_height = 256
        new_width = new_height * aspect_ratio
        
    image.thumbnail(size = [new_width, new_height])
    

    #Cropping
    crp_width = 224
    crp_height = 224
    
    width, height = image.size
    left = (width - crp_width)/2
    top = (height - crp_height)/2
    right = (width + crp_width)/2
    bottom = (height + crp_height)/2
    
    cropped_image = image.crop((left, top, right, bottom))

    #converting image to a numpy array and normalising it
    norm_image = np.array(cropped_image)/255 

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_image = (norm_image - means) /std
        
 
    np_image = norm_image.transpose(2, 0, 1)
    
    return np_image


def getFlowerName(labels, model, cat_to_name):
    ''' 
        Convert id to flower name.
    '''
    class_to_idx = model.class_to_idx
    idx_to_class = {value : key for (key, value) in class_to_idx.items()}        

    top_flowers = [cat_to_name[idx_to_class[label]] for label in labels]
    
    return  top_flowers


def predict(image_path, model, topk, device, cat_to_name):
    ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    if type(topk) == type(None):
        topk = 5
    else: 
        topk = topk
        
    input_tensor = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor)
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Find topk labels and their correspoding probability
    model.eval()
    logps = model.forward(input_tensor)
    prob = torch.exp(logps)
    
    top_probs, top_labels = prob.topk(topk)
     
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    
    top_flowers = getFlowerName(top_labels, model, cat_to_name)
    
    return top_probs, top_flowers
    

def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = load_model_from_checkpoint(args.checkpoint)
    
    device = getDevice(gpu_arg=args.gpu)
    
    top_probs, top_flowers = predict(args.image, model, args.top_k, device, cat_to_name)
    
    print(top_flowers, top_probs)

    
if __name__ == '__main__':
    main()