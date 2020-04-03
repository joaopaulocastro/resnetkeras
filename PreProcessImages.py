"""
    This is where we do image pre-processing steps, before training the model
"""

# imports
import numpy as np
import h5py
import os
from os import walk
import csv
import Constants as const
import ImageAsArray as myImg
import FindLabel as FindLabel
import datetime
import random

def PreProcessForDataGenerator():
    """
        This function goes through every image in the trainning set and does data augmentation

        The resulting trainning examples must be shuffled - otherwise, trainning will go through a series of 
    examples with the same label, one after another, and then model won't converge

        Also, the quantity of trainning examples for each class must be balanced (in other words, it's bad to have
    one class with a significantly higher quantity of examples than other). As our original set of images does not
    have this kind of balance, we solve that with data augmentation - classes that have fewer original images will
    generate more "artificial" examples.

        Trainning examples resulting from this procedure are stored in files, from where they can be later retrieved.

        The actual feed of examples to the model during trainning will be done through Data Generator
    """

    # read all images
    f_train = []
    for (dirpath, dirnames, filenames) in walk(const.TrainFolderPath):
        f_train.extend(filenames)
        break

    # load function to read label for each file
    fl = FindLabel.FindLabel(const.LabelsFilePath)

    # count images per class
    lbl_train = np.zeros((len(f_train)), dtype=np.uint8)
    X_train_qtde_por_classe = [0] * const.Y_Classes
    for i in range(0, len(f_train)):
        lbl_train[i] = fl.find_label(f_train[i]) - 1
        X_train_qtde_por_classe[lbl_train[i]] += 1
    print('Qty per class: ' + str(X_train_qtde_por_classe))

    # the quantity of original images per class will not be balanced
    # thus, before running the trainning, we "manually" count images per class (excel is our friend here)
    # then, we set a different quantity of rotations for each class - this is set at the constants file
    # that's the way I used to try to balance the quantity of examples per class

    # Total item count
    # batch count
    MultsPorImage = 1 * \
                len(const.HSteps) * \
                len(const.WSteps) * \
                len(const.filters) * \
                len(list(const.toneRange))
    X_train_Qtde = np.dot(X_train_qtde_por_classe, const.TrainSetRotations) * MultsPorImage
    batchCount = X_train_Qtde // const.BatchSize
    if (X_train_Qtde % const.BatchSize != 0):
        batchCount += 1

    # provision all sinthesized images for each original one
    AllExamples = [[None] * X_train_Qtde][0]
    curIndex = -1
    for i in range(0,len(f_train)):
        mult = const.TrainSetRotations[lbl_train[i]]
        ang = 360./mult

        for j in range(0,mult):
            for f in range(0,len(const.filters)):
                for deslocH in range(0,len(const.HSteps)):
                    for deslocW in range(0,len(const.WSteps)):
                        for tom in const.toneRange:
                            curIndex += 1
                            AllExamples[curIndex] = [f_train[i], 
                                lbl_train[i], 
                                j*ang, 
                                const.filters[f], 
                                const.HSteps[deslocH], 
                                const.WSteps[deslocW], 
                                tom*const.toneStep]

    # shuffle indexes
    indexes = np.arange(X_train_Qtde)
    np.random.shuffle(indexes)

    # folder where data will be saved
    # I've done many trainning iterations, each time using a different set of images and data augmentation parameters
    # at some point, I decided it would be a good idea to distinguish the folder where pre-processed data was stored...
    FolderPath = const.PreprocessedDataFolderPath(X_train_Qtde)
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath)

    # generate all chuncks of preprocessed data
    for b in range(0, batchCount):
        print('File ' + str(b+1))

        # how many items in this batch
        # comply with the last batch, which may have fewer items than the total allowed
        thisBatchItemCount = const.BatchSize
        if (b == (batchCount - 1)) and ((X_train_Qtde % const.BatchSize != 0)):
            thisBatchItemCount = X_train_Qtde % const.BatchSize

        # allocate arrays
        X_train = np.zeros(shape=(thisBatchItemCount, const.X_Height, const.X_Width, const.X_Channels), dtype=np.float16)
        Y_train = np.zeros(shape=(thisBatchItemCount, const.Y_Classes), dtype=np.uint8)

        # generate each item in the current chunk
        for i in range(0, thisBatchItemCount):
            thisIndex = indexes[(b*const.BatchSize) + i]

            # get file, label and rotation
            thisFile = AllExamples[thisIndex][0]
            thisLabel = AllExamples[thisIndex][1]
            thisDegree = AllExamples[thisIndex][2]
            thisFilter = AllExamples[thisIndex][3]
            thisStepH = AllExamples[thisIndex][4]
            thisStepW = AllExamples[thisIndex][5]
            thisTom = AllExamples[thisIndex][6]

            # recupera a image
            X_train[i] = myImg.GetImagePreprocessedArray(const.TrainFolderPath + thisFile,
                                                        filter=thisFilter,
                                                        tone=thisTom,
                                                        stepH=thisStepH,
                                                        stepW=thisStepW,
                                                        degree=thisDegree)[:,:,0:3]
            Y_train[i, thisLabel] = 1

        # save current chunk to file
        f = h5py.File(FolderPath + 'XY' + str(b+1) + '.hdf5', "w")
        f.create_dataset('X', data=X_train)
        f.create_dataset('Y', data=Y_train)
        f.close()

    print("Loaded trainning set: ")
    print('Qty: ' + str(X_train_Qtde))
    print('Batches: ' + str(batchCount))

    return X_train_Qtde, batchCount