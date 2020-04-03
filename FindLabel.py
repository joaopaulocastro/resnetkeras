"""
    This is a class that loads the labels to memory and then makes a function available to search for a specific label
    
    Our dataset has (relatively) few images
    for each image in the original dataset, we will use data augmentation to generate new trainning samples
    so, each original image will be used to create many, many "artificial" images

    then, it's efficient to load the whole labels file into memory to search for labels
"""

# imports
import csv

class FindLabel():
    def __init__(self, labelsFile):
        # load labels from file
        with open(labelsFile, newline='') as csvfile:
            self.my_labels = list(csv.reader(csvfile, delimiter=';'))

    # function to fetch label
    def find_label(self, filename, raiseError = True):
        for sublist in self.my_labels:
            if sublist[0] in filename:
                return int(sublist[1])
        
        if raiseError:
            raise ValueError('No label found' + filename)
        else:
            return -1