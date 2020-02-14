import pandas as pd
import numpy as np
from prep import dataset
from densenet import DenseNet
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.optimizers import RMSprop
file_path = 'content/parrot_proj1/scoring/'


def run_model():
    x_train, y_train = dataset.train()
    y_train = np_utils.to_categorical(y_train, 6)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=0.2)
    print("x_train shape : ", x_train.shape)
    print("x_valid shape : ", x_valid.shape)
    print("y_valid shape : ", y_train.shape)
    print("y_valid shape : ", y_valid.shape)
    model = DenseNet(nb_blocks=4, nb_filters=128, depth=40,
                     growth_rate=12, compression=0.2, input_shape=(150, 150),
                     )
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=2e-5))
    model.summary()
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            # zoom_range = 0.2, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                               steps_per_epoch=x_train.shape[0]//64, epochs=20,
                               validation_data=(x_valid, y_valid))
    return model, hist
