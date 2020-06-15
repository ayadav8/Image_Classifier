# PROGRAMMER: Amit Yadav
# DATE CREATED: 06/06/2020

# Imports python modules

from get_input_args import get_input_args
from functions import train_transforms,valid_transforms,test_transforms
from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from workspace_utils import keep_awake

def main():
    
    ###Get input from user
    in_arg = get_input_args()
    print(in_arg)
    
    train_dir = in_arg.dir + '/train'
    valid_dir = in_arg.dir + '/valid'
    test_dir = in_arg.dir + '/test'
    
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder( train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder( valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder( test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    ####Define the model
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    
    models_dic = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet':densenet121}
    
    model_name = in_arg.arch
    
    ###Load the userdefined model
    model = models_dic[model_name]
    
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
    
    epochs = in_arg.epocs
    steps = 0
    running_loss = 0
    print_every = 5

    for i in keep_awake(range(5)):
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
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(testloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()
    
    ########Save the model
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'input_size': 1024,
                   'output_size': in_arg.output_nodes,
                    'hidden_layers': [each for each in model.classifier],
                    'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, in_arg.save_dir)
       
# Call to main function to run the program
if __name__ == "__main__":
    main()