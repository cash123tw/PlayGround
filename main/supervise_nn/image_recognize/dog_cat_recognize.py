import os.path
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import main.utils.downloader as downloader

from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from main.utils.plot import make_img

data_set_name = 'tongpython/cat-and-dog'
data_set_dir = os.path.join('../../../data/dog_cat')


def load_dog_cat_data(batch_size, size):
    sedd = 123
    train_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(os.path.join(data_set_dir, 'training_set/training_set'),
                                                   target_size=size,
                                                   batch_size=batch_size, shuffle=True, seed=sedd, subset='training')
    valid_data = test_datagen.flow_from_directory(os.path.join(data_set_dir, 'test_set/test_set'), target_size=size,
                                                  batch_size=batch_size, shuffle=True, seed=sedd, subset='training')
    return train_data, valid_data


# vgg16 model
def vgg16_model(input_shape, output_shape) -> tf.keras.Model:
    input_layer = vgg19.VGG19(include_top=False, input_shape=input_shape)
    dense_layer = layers.Dense(output_shape, activation="sigmoid")
    model = tf.keras.Model(input_layer.inputs, dense_layer(input_layer.output))
    return model


# 參考別人設計
def cnn_model(input_shape, output_shape) -> keras.models.Sequential:
    model = keras.models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output_shape, activation='sigmoid'))
    return model


def vgg19_model(input_shape, output_shape) -> keras.Model:
    base_model = vgg19.VGG19(include_top=False)
    base_model.trainable = False

    global_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(output_shape)

    inputs = tf.keras.Input(shape=input_shape)
    outputs = vgg19.preprocess_input(inputs)
    outputs = base_model(outputs)
    outputs = global_layer(outputs)
    outputs = prediction_layer(outputs)

    model = tf.keras.Model(inputs, outputs)

    return model


def dog_cat_main(model_path=None, load_model=False, lr=1e-4):
    train_data, valid_data = load_dog_cat_data(32, (224, 224))

    if load_model and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # model = vgg19_model(train_data.image_shape, train_data.num_classes)
        # model = vgg16_model(train_data.image_shape, train_data.num_classes)
        model = cnn_model(train_data.image_shape, train_data.num_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy'],
            loss=keras.losses.CategoricalCrossentropy()
        )
        model.summary()

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='min')

    record = model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[early_stop_callback])
    history = record.history

    make_img('track','loss','accuracy',history['loss'],history['accuracy'],
             save_path=os.getcwd(),save_name='loss_accuracy.jpg',show=True)

    model.save(model_path)


if __name__ == '__main__':
    if not os.path.exists(data_set_dir):
        downloader.download_kaggle_file(data_set_name, data_set_dir)
    dog_cat_main(os.path.join('../../../lib/dog_cat_classification_model'))
