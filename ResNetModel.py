"""
    This module is my exercise on understanding ResNet architecture while manually building the model,
    or, alternativelly, loading the architecture bult into keras library

    The keras reference model I used can be located at 
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py

"""

# imports
import Constants as const
import tensorflow as tf

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform


### =============== my implementation for ResNet architectures : BEGIN ============================

def identity_block(X, f, filters, stage, block, small = False):
    """
    Identity block implementation

    arguments:
        X = input tensor resulting from previous block, having shape(m, H_prev, W_prev, C_prev), where:
            m 
            H_prev = H resulting from previous block
            W_prev = W resulting from previous block
            C_prev = Channels resultring from previous block
        f = integer, shape of the middle convolution window for the main path (kernel size)
        filters = integer list, number of filters in the convolution layers for the main path
        stage = integer used to give each layer a different name
        block = string used to give each layer a different name
        small = used to differentiate kernel size and padding when building Small or Large resnet
    
    Returns:
        X -- output of the identity block, tensor of shape (H, W, C)
    """

    # give each layer a different name
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value.  
    X_shortcut = X

    # kernel size and padding for 1st components
    ks = (1, 1)
    pad = 'valid'
    if (small):
        ks = (f, f)
        pad = 'same'
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = ks, strides = (1,1), padding = pad, name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    if (not small):
        X = Activation('relu')(X)
        
        # Third component of main path 
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = layers.Add()([X_shortcut, X])  
    X = Activation('relu')(X)
    
    return X

def conv_block(X, f, filters, stage, block, s = 2, small = False):
    """
    Implementation of the convolutional block 
    
    Arguments:
        X = input tensor resulting from previous block, having shape(m, H_prev, W_prev, C_prev), where:
            m 
            H_prev = H resulting from previous block
            W_prev = W resulting from previous block
            C_prev = Channels resultring from previous block
        f = integer, shape of the middle convolution window for the main path (kernel size)
        filters = integer list, number of filters in the convolution layers for the main path
        stage = integer used to give each layer a different name
        block = string used to give each layer a different name
        s = integer, defines stride
        small = used to differentiate kernel size and padding when building Small or Large resnet
    
    Returns:
        X -- output of the convolutional block, tensor of shape (H, W, C)
    """
    
    # give each layer a different name
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # kernel size and padding for 1st components
    ks = (1, 1)
    pad = 'valid'
    shortcutFilters = F3
    if (small):
        ks = (f, f)
        pad = 'same'
        shortcutFilters = F2

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, ks, strides = (s,s), padding = pad, name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    if (not small):
        X = Activation('relu')(X)

        # Third component of main path 
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = shortcutFilters, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    ##### END SHORTCUT PATH #### 

    # Final step: RELU activation 
    X = layers.Add()([X_shortcut, X]) 
    X = Activation('relu')(X)
    
    return X

def ResNetSmall(input_shape = (const.X_Height, const.X_Width, const.X_Channels), classes = const.Y_Classes, Layers = 18):
    """
    Implementation of small size ResNet (18 or 34 layers) 
    """
    if (Layers != 18) and (Layers != 34):
        raise ValueError('Invalid layer count: ' + str(Layers) + " (must be 18 or 34).")
    Layers34 = (Layers == 34)

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = conv_block(X, f = 3, filters = [64, 64, None], stage = 2, block='a', s = 1, small = True)
    X = identity_block(X, 3, [64, 64, None], stage=2, block='b', small = True)
    if (Layers34):
        X = identity_block(X, 3, [64, 64, None], stage=2, block='c', small = True)

    # Stage 3
    X = conv_block(X, f = 3, filters = [128, 128, None], stage = 3, block='a', s = 2, small = True)
    X = identity_block(X, 3, [128, 128, None], stage=3, block='b', small = True)
    if (Layers34):
        X = identity_block(X, 3, [128, 128, None], stage=3, block='c', small = True)
        X = identity_block(X, 3, [128, 128, None], stage=3, block='d', small = True)

    # Stage 4 
    X = conv_block(X, f = 3, filters = [256, 256, None], stage = 4, block='a', s = 2, small = True)
    X = identity_block(X, 3, [256, 256, None], stage=4, block='b', small = True)
    if (Layers34):
        X = identity_block(X, 3, [256, 256, None], stage=4, block='c', small = True)
        X = identity_block(X, 3, [256, 256, None], stage=4, block='d', small = True)
        X = identity_block(X, 3, [256, 256, None], stage=4, block='e', small = True)
        X = identity_block(X, 3, [256, 256, None], stage=4, block='f', small = True)

    # Stage 5
    X = conv_block(X, f = 3, filters = [512, 512, None], stage = 5, block='a', s = 2, small = True)
    X = identity_block(X, 3, [512, 512, None], stage=5, block='b', small = True)
    if (Layers34):
        X = identity_block(X, 3, [512, 512, None], stage=5, block='c', small = True)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet' + str(Layers))

    return model

def ResNetLarge(input_shape = (const.X_Height, const.X_Width, const.X_Channels), classes = const.Y_Classes, Layers = 50, weights = 'imagenet'):
    """
    Implementation of large size ResNet (50, 101 or 152 layers) 
    """

    # validate parameters
    if (Layers != 50) and (Layers != 101) and (Layers != 152):
        raise ValueError('Invalid layer number: ' + str(Layers) + " (must be 50, 101 or 152).")

    if (weights not in [None, 'imagenet']):
        raise ValueError('Invalid weights definition: ' + str(weights) + " (must be None or 'imagenet'.")

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = conv_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    stage3Loops = 3
    if (Layers == 152):
        stage3Loops = 7

    X = conv_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    for i in range(0, stage3Loops):
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='b' + str(i))

    # Stage 4
    stage4Loops = 5
    if (Layers == 101):
        stage4Loops = 22
    elif (Layers == 152):
        stage4Loops = 35

    X = conv_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    for i in range(0, stage4Loops):
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    # Stage 5
    X = conv_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Create model
    model_name = 'resnet' + str(Layers)
    model = Model(inputs = X_input, outputs = X, name=model_name)

    # time to load weights (if they were required)
    if (weights == 'imagenet'):
        BASE_WEIGHTS_PATH = (
            'https://github.com/keras-team/keras-applications/'
            'releases/download/resnet/')
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = tf.keras.utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models')
        model.load_weights(weights_path, by_name=False)

    return model

### =============== END my implementation for ResNet architectures  ============================

def ResNet(input_shape = (const.X_Height, const.X_Width, const.X_Channels), classes = const.Y_Classes, Layers = 50, source = 'keras', weights = 'imagenet'):
    """
    create RestNet model

    Arguments:
        input_shape = Height, Width and channels for each input image
        classes = how many classes model will be trainned on
        Layers: how many layers; should be one of [18, 34, 50, 101, 152]
        source: 'keras' (use built-in model) or 'manual' (use my custom model above)
        weights: 'imagenet' (load weights from keras lib) or None (no weights loading) 
            'imagenet' only available if layers in [50,101,152]
    """

    # validate parameters
    if (Layers not in [18, 34, 50, 101, 152]):
        raise ValueError('Invalid layer number: ' + str(Layers) + ' (must be one of [18, 34, 50, 101, 152]).')

    if (source not in ['keras', 'manual']):
        raise ValueError('Invalid model source: ' + str(source) + " (must be 'keras' or 'manual'.")

    if (weights not in [None, 'imagenet']):
        raise ValueError('Invalid weights definition: ' + str(weights) + " (must be None or 'imagenet'.")

    if (Layers in [18, 34]):
        if (source == 'keras'):
            raise ValueError("No keras model available for small ResNets. 'source' parameter must be 'manual' when layers are 18 or 34.")

        if (weights != None):
            raise ValueError("No weights available for small ResNets. 'weights' Parameter must be None when layers are 18 or 34.")

    # build model
    if (source == 'keras'):
        # load base model from keras
        if (Layers == 50):
            from keras.applications.resnet import ResNet50
            baseModel = ResNet50(include_top = False, weights = weights, input_shape = input_shape)
        elif (Layers == 101):
            from keras.applications.resnet import ResNet101
            baseModel = ResNet101(include_top = False, weights = weights, input_shape = input_shape)
        elif (Layers == 152):
            from keras.applications.resnet import ResNet152
            baseModel = ResNet152(include_top = False, weights = weights, input_shape = input_shape)
    elif (source == 'manual'):
        # load model from my implementation
        if (Layers in [18,34]):
            baseModel = ResNetSmall(input_shape=input_shape, classes=classes, Layers=Layers)
        else:
            baseModel = ResNetLarge(input_shape=input_shape, classes=classes, Layers=Layers, weights=weights)

    # add final layers to built-in keras model
    from keras.models import Model
    from keras.layers import Dense, Flatten, AveragePooling2D

    X = baseModel.output
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    X = Flatten()(X)
    Preds = Dense(const.Y_Classes, activation='softmax', name='fc' + str(const.Y_Classes))(X)

    model = Model(inputs=baseModel.input, outputs=Preds)

    # return the model
    return model

def Compile(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def get_model_memory_usage(model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*(shapes_mem_count + trainable_count + non_trainable_count)
    # gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    # return gbytes
    return total_memory

