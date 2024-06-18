import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, Reshape,UpSampling2D, MaxPooling2D, Concatenate, Dense, Flatten, Activation, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import load_dataset

def conv_layer(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation=None):
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(inputs)

def brelu(x):
    return tf.maximum(0.0, tf.minimum(1.0, x))

class Maxout(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_units = num_units

    def call(self, inputs):
        shape = tf.shape(inputs)
        num_channels = shape[-1]
        tf.debugging.assert_equal(num_channels % self.num_units, 0, message="Number of channels must be divisible by num_units")
        new_shape = tf.concat([shape[:-1], [self.num_units, num_channels // self.num_units]], axis=-1)
        output = tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)
        return output

def dehazenet(input_shape):
    input = Input(shape=input_shape)

    # Feature Extraction
    x = conv_layer((input), filters=16, kernel_size=(5, 5), activation='relu')
    
    # Feature Extraction Maxout
    x = Maxout(4)(x)

    # Multi-Scale Processing
    
    branch1 = conv_layer((x), filters=16, kernel_size=(3, 3), activation='relu')
    branch2 = conv_layer(x, filters=16, kernel_size=(5, 5), activation='relu')
    branch3 = conv_layer(x, filters=16, kernel_size=(7, 7), activation='relu')

    # Concatenation
    x = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Local Extremum
    x = MaxPooling2D(pool_size=(7, 7))(x)

    # Non-linear Regression
    x = conv_layer(x, filters=48, kernel_size=(6, 6))
    x = Activation(brelu)(x)

    x= UpSampling2D(size=(7, 7))(x)
    x=Conv2DTranspose(filters=16, kernel_size=(7, 7), padding='same')(x)
    x = Conv2DTranspose(filters=input_shape[-1], kernel_size=(3, 3), padding='same')(x)
    model = Model(inputs=input, outputs=x, name="DehazeNet")
    
    return model

def train_model(input_shape, input_dir, target_dir):
    # Load preprocessed data
    X, y = load_dataset(input_dir, target_dir, size=input_shape[:2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = dehazenet(input_shape)

    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    model.save('D:\\image-dehazing\\model\\models\\dehazing_model.h5')

if __name__ == '__main__':
    input_shape = (196, 196, 3)
    input_dir = 'D:\\image-dehazing\\model\\data\\train\\input'
    target_dir = 'D:\\image-dehazing\\model\\data\\train\\target'

    train_model(input_shape, input_dir, target_dir)
