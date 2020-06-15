# PROGRAMMER: Amit Yadav
# DATE CREATED: 06/06/2020

# Imports python modules
import argparse


def get_input_args():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--dir', type=str, default='flowers', 
                        help='path to folder of images')
   # TODO: 1a. EDIT parse.add_argument statements BELOW to add type & help for:
    #          --arch - the CNN model architecture
    #          --dogfile - text file of names of dog breeds
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'the model architecture')
    parser.add_argument('--labelfile', type = str, default = 'cat_to_name.json', help = 'path to label file' )
    parser.add_argument('--learning_rate', type = str, default = 0.003, help = 'learning rate' )
    parser.add_argument('--hidden_nodes', type = str, default = 500, help = 'hidden nodes' )
    parser.add_argument('--output_nodes', type = str, default = 102, help = 'output nodes' )
    parser.add_argument('--epocs', type = str, default = 1, help = 'number of epocs' )
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'path to save the model' )
    parser.add_argument('--path_image', type = str, default = './flowers/test/102/', help = 'path to the image for inference' )
    parser.add_argument('--top_prb', type = int, default = 5, help = 'top probabilities to show' )
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
