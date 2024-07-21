import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import load_dataset
from tensorflow import keras

# Clear any custom objects to avoid conflicts
keras.saving.get_custom_objects().clear()

def conv_layer(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation=None):
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(inputs)

def dehazenet(input_shape):
    input = Input(shape=input_shape)
    print("Input Layer", input.shape)

    # conv1
    conv1 = conv_layer(input, filters=20, kernel_size=(5, 5), activation=None)
    relu1 = ReLU()(conv1)
    print("Conv1 Layer", relu1.shape)

    # pool1
    pool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(relu1)
    print("Pool1 Layer", pool1.shape)

    # conv2/1x1, conv2/3x3, conv2/5x5, conv2/7x7
    conv2_1x1 = conv_layer(pool1, filters=16, kernel_size=(1, 1), activation=None)
    conv2_3x3 = conv_layer(pool1, filters=16, kernel_size=(3, 3), padding='same', activation=None)
    conv2_5x5 = conv_layer(pool1, filters=16, kernel_size=(5, 5), padding='same', activation=None)
    conv2_7x7 = conv_layer(pool1, filters=16, kernel_size=(7, 7), padding='same', activation=None)

    # concat conv2/output
    conv2_output = Concatenate(axis=-1)([conv2_1x1, conv2_3x3, conv2_5x5, conv2_7x7])
    relu2 = ReLU()(conv2_output)
    print("Concat and ReLU2 Layer", relu2.shape)

    # pool2
    pool2 = MaxPooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(relu2)
    print("Pool2 Layer", pool2.shape)

    # ip1
    ip1 = Conv2D(filters=48, kernel_size=(1, 1))(pool2)
    drelu1 = ReLU()(ip1)
    print("ip1 Layer", drelu1.shape)

    # Upsampling
    upsample = UpSampling2D(size=(1, 1))(drelu1)
    conv2dtranspose1 = Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same')(upsample)
    output = Conv2DTranspose(filters=input_shape[-1], kernel_size=(3, 3), padding='same')(conv2dtranspose1)
    print("Output Layer", output.shape)

    model = Model(inputs=input, outputs=output, name="DehazeNet")
    return model

def train_model(input_shape, input_dir, target_dir):
    # Load preprocessed data
    X, y = load_dataset(input_dir, target_dir, size=input_shape[:2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = dehazenet(input_shape)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    
    # Early stopping callback to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
    model.save('D:\\image-dehazing\\model\\models\\dehazing_model.keras')
    
    # Load the saved model and verify
    reconstructed_model = keras.models.load_model("D:\\image-dehazing\\model\\models\\dehazing_model.keras")
    np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))

if __name__ == '__main__':
    input_shape = (144, 144, 3)
    input_dir = 'D:\\image-dehazing\\model\\data\\train\\input'
    target_dir = 'D:\\image-dehazing\\model\\data\\train\\target'

    train_model(input_shape, input_dir, target_dir)
