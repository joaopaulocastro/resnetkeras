"""
    This is the main file which implements the processing pipeline going through the top-level steps
"""

# imports
import keras.backend as K
import PreProcessImages as pre
import ResNetModel as rn
import DataGenerator as dg
import Constants as const

# configure keras
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# pre-processes trainning set
examples, batches = pre.PreProcessForDataGenerator()

# load desired model
# here we could play with parameters, testing different architecture sizes, transfer learning, etc...
# the idea is to see which model provides the best cost/benefit, adjust hyper-parameters, etc
layers = 101
source = 'keras'
weights = 'imagenet'
model = rn.ResNet(Layers = layers, source = source, weights = weights)

# compiles model
rn.Compile(model)
model.save(const.SavedModelFileName(layers, 0))

# generator
trainGenerator = dg.DataGeneratorFromPreprocessed(examples, batches)

# epochs
epochs = 5

# train through several epochs
for i in range(0, epochs+1):
    model.fit_generator(generator=trainGenerator, epochs=1)
    model.save(const.SavedModelFileName(layers, i))

# do batch evaluation
import BatchPredict as bp
import LoadSets as load
from keras.models import load_model

for i in range(0, epochs+1):
    # load model 
    print("will load model")
    model = load_model(const.SavedModelFileName(layers, i))
    print("loaded model ")

    Qtde, Files, X, Y = load.LoadSet(const.TestFolderPath)
    bp.batchEvaluate(model, "test" + str(i), Qtde, Files, X, Y)

    Qtde, Files, X, Y = load.LoadSet(const.TrainFolderPath)
    bp.batchEvaluate(model, "train" + str(i), Qtde, Files, X, Y)