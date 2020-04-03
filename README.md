# Image Classification pipeline with ResNet architecture implemented in Keras (+Python)

This is an exercise in building a pipeline for training a model to do image classification.

The chosen CNN architecture was ResNet, and we experimented with several network depths: 18, 34, 50, 101 and 152 layers. 

We've experimented manually building the layered architecture, and also using the model built into keras library.

There is an option of loading pre-trained weights to achieve Transfer Learning.

The images are acquired from cameras with fish-eye lenses, so they are circular. Each image contains zero or exactly one object. So, the classification classes are:
.0: None (no object)
.1: car
.2: bus
.3: truck
.4: bike

Our original dataset is small (around 1.100 image samples), so we exercised doing Data Augmentation using a series of combined "silly" modifications over each original image, such as rotating, shifting some pixels on the height and the width and applying Image filters.

The pipeline has the following steps:
1. pre-process original dataset, doing data augmentation and saving training examples to files
2. build and compile resnet architecture; load pre-trained weights (if transfer learning is desired)
3. create data generator to supply data to the model fitting process
4. fit the model over a series of epochs; save model to a file at the end of each epoch
5. do batch prediction using the saved trained models
6. use a spreadsheet to compare metrics (Precision, Recall, F1 score) for each saved model
