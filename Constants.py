# ####################### input size
X_Width = 128
X_Height = 128
X_Channels = 3

# ####################### possible classes
Y_Classes = 5   # 0 = none, 1 = car, 2 = bus, 3 = truck, 4 = bike

# ####################### batch size = how many examples submitted in each training batch
BatchSize = 64

# ####################### paths to required folders
LabelsFilePath = 'data/labels.csv'
TrainFolderPath = 'data/images/train/'
TestFolderPath = 'data/images/test/'
BatchEvalFolderPath = 'data/batchEval/'

# ####################### functions that define paths based on parameters
def PreprocessedDataFolderPath(ExampleCount):
    basePath = 'data/preprocessed/'
    return basePath + str(X_Width) + 'x' + str(X_Height) + 'x' + str(X_Channels) + 'x' + str(ExampleCount) + '/'

def SavedModelFileName(layers, epochs):
    CurrentPath = 'data/savedModel/'
    CurrentFile = 'Resnet_' + str(layers) + "_" + str(X_Height) + 'x' + str(X_Width) + 'x' + str(X_Channels) + '_' + str(epochs) + 'epochs.h5'
    return CurrentPath + CurrentFile

# ####################### data augmentation parameters
# - different quantity of rotations for each class
TrainSetRotations = [1, 1, 11, 13, 2]

# - image shifts, on the height (H) and width (W)
HSteps = [-10, 0, 10]
WSteps = [-10, 0, 10]

# - filters
from PIL import ImageFilter
filters = [None, 
            ImageFilter.BLUR, 
            ImageFilter.MaxFilter, 
            ImageFilter.ModeFilter]

# tone (integer do add/subtract to each RGB pixel value)
toneStep = 20
toneRange = range(-1,2)