"""
    This is a batch predction procedure

    As our goal is to evaluate many experiments, varying hyper parameters, to decide which will provide the best
cost/benefit, we will use this procedure in "batch" mode to evaluate the many models an then load the results

    Loading the results in spreadsheets (hey, Excel!) will allow us to evaluate Precision, Recall and F1 score
"""

# imports
import tensorflow as tf
import numpy as np
from keras.models import Model, load_model
import Constants as const

def batchEvaluate(model, prefix, Qtde, Files, X_set, Y_set, modelChannels = const.X_Channels):
    if modelChannels == const.X_Channels:
        X = X_set
    else:
        shape = (X_set.shape[0], X_set.shape[1], X_set.shape[2], modelChannels)
        X = np.zeros(shape, np.float16)
        X[:,:,:,:] = X_set

    # provision an array to hold the results
    # it will have 4 columns:
    #   - col #0: file name
    #   - col #1: Y (that is, the label)
    #   - col #2: Y' (that is, the prediction)
    #   - col #3: prediction accuracy
    eval_result = np.empty(shape=(Qtde, 4), dtype=np.object)

    for i in range(0, Qtde):
        eval_result[i,0] = Files[i]
        eval_result[i,1] = np.argmax(Y_set[i])

        pred = model.predict(X[i:i+1,:,:,:])
        index = np.argmax(pred)
        eval_result[i,2] = index
        eval_result[i,3] = pred[0,index] * 100

        print(eval_result[i][1:5])

    np.savetxt(const.BatchEvalFolderPath + '/' + prefix + ".csv", eval_result, delimiter=";", fmt="%s")
    print("saved " + prefix + ".csv")

