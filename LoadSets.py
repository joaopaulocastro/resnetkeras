"""
    Procedure to load data from trainning/test sets
"""

# imports
import Constants as const
import FindLabel
import ImageAsArray as myImg
import numpy as np
from os import walk

def LoadSet(folder, classFile = const.LabelsFilePath):
    
    # get all files
    f = []
    for (dirpath, dirnames, filenames) in walk(folder):
        f.extend(filenames)
        break

    fl = FindLabel.FindLabel(classFile)

    lbl = np.zeros((len(f)), dtype=np.uint8)

    for i in range(0, len(f)):
        l = fl.find_label(f[i], raiseError=False)
        if l < 0:
            l = const.Y_Classes + 1
        lbl[i] = l - 1

    # prepare training set
    Qtde = len(f)
    X = np.empty(shape=(Qtde, const.X_Height, const.X_Width, const.X_Channels), dtype=np.float16)
    Y = np.zeros(shape=(Qtde, const.Y_Classes + 1), dtype=np.uint8)

    # print('X.nbytes: ' + str(X.nbytes))

    for i in range(0, len(f)):
        if (i % 50) == 0:
            print(i)

        X[i] = myImg.GetImagePreprocessedArray(folder + '/' + f[i])
        Y[i, lbl[i]] = 1

    print("Loaded set: ")
    print('Qtde: ' + str(Qtde))

    return Qtde, f, X, Y