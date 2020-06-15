import matplotlib.pyplot as plt
from functions import predict 
from get_input_args import get_input_args
from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
import os, random
from PIL import Image
import json
import re

def main():
    
    in_arg = get_input_args()
    
    
    ####Define the model
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    
    models_dic = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet':densenet121}
    
    model_name = in_arg.arch
    
    ###Load the userdefined model
    model = models_dic[model_name]
    
    ###Load label file
    with open(in_arg.labelfile, 'r') as f:
        cat_to_name = json.load(f)
    
    # Freeze parameters so we don't backprop through them
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, in_arg.hidden_nodes)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(in_arg.hidden_nodes, in_arg.output_nodes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device);
   
    
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer.load_state_dict(checkpoint['optimizer'])                                  
        model.load_state_dict(checkpoint['state_dict'])

        return model, optimizer

    ###Load the saved model
    model, optimizer = load_checkpoint(in_arg.save_dir)
       
    # TODO: Display an image along with the top x classes
    img = random.choice(os.listdir(in_arg.path_image))
    img_path = in_arg.path_image + img
    
    ##Print the actual label of the image
    key = re.findall("\d+", in_arg.path_image)
    print("Actual image label:{}".format(cat_to_name[key[0]]))

    prob, classes = predict(img_path, model)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])
    
# Call to main function to run the program
if __name__ == "__main__":
    main()