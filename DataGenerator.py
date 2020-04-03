"""
    This is the class used to provide trainning examples to the model in real time, during trainning

    Is this implementation, it gets data from pre-processed files (resulting of the pre-processing and data augmentation steps)
"""

# imports
import keras
import Constants as const
import numpy as np
import h5py

class DataGeneratorFromPreprocessed(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, QtdeExamples, QtdeBatches, QtdeChannels = const.X_Channels):
        'Initialization'
        # folder where to fetch data
        self.folderPath = const.PreprocessedDataFolderPath(QtdeExamples)
        self.qtdeBatches = QtdeBatches
        self.qtdeChannels = QtdeChannels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.qtdeBatches

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.qtdeBatches - 1)
        np.random.shuffle(self.indexes)
        self.indexes = np.append(self.indexes, self.qtdeBatches - 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation(index)

        return X, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples'
        i = self.indexes[index]
        f = h5py.File(self.folderPath + 'XY' + str(i+1) + '.hdf5', "r")
        x0 = f['X'][:]
        Y = f['Y'][:]
        f.close()

        if self.qtdeChannels == const.X_Channels:
            X = x0
        else:
            shape = (x0.shape[0], x0.shape[1], x0.shape[2], self.qtdeChannels)
            X = np.zeros(shape, np.float16)
            X[:,:,:,:] = x0

        return X, Y
