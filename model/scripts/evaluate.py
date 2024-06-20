import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, Accuracy
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import load_dataset
from tensorflow import keras
keras.saving.get_custom_objects().clear()
# Define and register brelu as an activation function

# Define and register Maxout custom layer
@keras.saving.register_keras_serializable(package="MyLayers")
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

    def get_config(self):
        config = super(Maxout, self).get_config()
        config.update({"num_units": self.num_units})
        return config

@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def brelu(x):
    return tf.maximum(0.0, tf.minimum(1.0, x))

def evaluate_model(model_path, input_shape, input_dir, target_dir):
    # Load preprocessed data
    X, y = load_dataset(input_dir, target_dir, size=input_shape[:2])

    # Split the data into training and testing sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model(model_path)

    # Evaluate the model
    mse = MeanSquaredError()
    accuracy = Accuracy()

    y_pred = model.predict(X_test)
    mse.update_state(y_test, y_pred)
    accuracy.update_state(y_test, np.argmax(y_pred, axis=-1))

    mse_result = mse.result().numpy()
    accuracy_result = accuracy.result().numpy()

    print(f'Mean Squared Error: {mse_result}')
    print(f'Accuracy: {accuracy_result}')

if __name__ == '__main__':
    model_path = 'D:\\image-dehazing\\model\\models\\dehazing_model.keras'
    input_shape = (119, 119,3)
    input_dir = 'D:\\image-dehazing\\model\\data\\train\\input'
    target_dir = 'D:\\image-dehazing\\model\\data\\train\\target'

    evaluate_model(model_path, input_shape, input_dir, target_dir)
