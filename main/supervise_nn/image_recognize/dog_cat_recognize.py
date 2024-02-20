import os.path
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import main.utils.downloader as downloader
import matplotlib.pyplot as plt

from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_set_name = 'tongpython/cat-and-dog'
data_set_dir = os.path.join('../../../data/dog_cat')

class ModelType(Enum):
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    CUSTOM1 = 'custom_cnn'

def load_dog_cat_data(batch_size, size):
    sedd = 123
    train_datagen = ImageDataGenerator(horizontal_flip=True)
    test_datagen = ImageDataGenerator()

    train_data = train_datagen.flow_from_directory(os.path.join(data_set_dir, 'training_set/training_set'),
                                                   target_size=size,
                                                   batch_size=batch_size, shuffle=True, seed=sedd, subset='training')
    valid_data = test_datagen.flow_from_directory(os.path.join(data_set_dir, 'test_set/test_set'), target_size=size,
                                                  batch_size=batch_size, shuffle=True, seed=sedd, subset='training')
    return train_data, valid_data


# vgg16 model
def vgg16_model(input_shape, output_shape) -> tf.keras.Model:
    base_model = vgg16.VGG16(include_top=False, input_shape=input_shape)

    inputs = layers.Input(shape=input_shape)

    outputs = vgg16.preprocess_input(inputs)
    outputs = base_model(outputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(512, activation="relu")(outputs)
    outputs = layers.Dense(256, activation="relu")(outputs)
    outputs = layers.Dense(output_shape, activation="softmax")(outputs)

    model = tf.keras.Model(inputs, outputs)
    return model


# 參考別人設計
def cnn_model(input_shape, output_shape) -> keras.models.Sequential:
    model = keras.models.Sequential()
    model.add(layers.Lambda(lambda x:tf.divide(x,255),input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (3, 3), 1, activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(256, (3, 3), 1, activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model


def vgg19_model(input_shape, output_shape) -> keras.Model:
    base_model = vgg19.VGG19(include_top=False)
    # base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = vgg19.preprocess_input(inputs)
    x = base_model(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(output_shape,activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def dog_cat_main(model_path=None, load_model=False, lr=1e-4, model_type:ModelType = ModelType.VGG16):
    train_data, valid_data = load_dog_cat_data(32, (224, 224))

    if load_model and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        #select model
        if model_type == ModelType.VGG16:
            model = vgg16_model(train_data.image_shape, train_data.num_classes)
        elif model_type == ModelType.VGG19:
            model = vgg19_model(train_data.image_shape, train_data.num_classes)
        else:
            model = cnn_model(train_data.image_shape, train_data.num_classes)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy'],
            loss=keras.losses.CategoricalCrossentropy()
        )
        model.summary()

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.05, patience=5, mode='min')

    record = model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[early_stop_callback])
    history = record.history

    model.save(model_path)

    #show track history
    track_list = ['loss','accuracy','val_loss','val_accuracy']
    for key in track_list:
        plt.plot(history[key])

    plt.xlabel('epochs')
    plt.legend(track_list,loc='upper left')
    plt.savefig(os.path.join(os.getcwd(),f'track_{model_type.value}.jpg'))
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(data_set_dir):
        downloader.download_kaggle_file(data_set_name, data_set_dir)

    dog_cat_main(os.path.join('../../../lib/dog_cat_classification_model'),model_type=ModelType.CUSTOM1)
    # for i in [ModelType.CUSTOM1,ModelType.VGG16,ModelType.VGG19]:
    #     dog_cat_main(os.path.join('../../../lib/dog_cat_classification_model'),model_type=i)

