from torchvision import datasets, transforms, models
import torch
from PIL import Image
import numpy as np

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    sample_image = Image.open(image)
    #Image resize
    sample_image = sample_image.resize((256,256))
    
    #Image centre crop
    s = 0.5*(256-224)
    sample_image = sample_image.crop((s,s,256-s,256-s))
    
    #Image to array
    sample_image = np.array(sample_image)/255
    
    #Image normalize
    sample_image = (sample_image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    
    #Image transpose
    sample_image = sample_image.transpose(2,0,1)
    
    return sample_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
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

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    output = model.forward(image)
    ps = torch.exp(output).data
    
    top_p = torch.topk(ps, topk)[0].tolist()[0] 
    top_class = torch.topk(ps, topk)[1].tolist()[0]
    
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    label = []
    for i in range(5):
        label.append(ind[top_class[i]])

    return top_p, label