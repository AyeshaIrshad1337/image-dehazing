import tensorflow as tf
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from preprocessing import load_dataset
from tensorflow import keras

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

# Define and register brelu as an activation function
@keras.saving.register_keras_serializable(package="MyActivations", name="brelu")
def brelu(x):
    return tf.maximum(0.0, tf.minimum(1.0, x))

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

def evaluate_model(model_path, input_dir, target_dir, size=(256, 256)):
    """
    Evaluate the trained model on the test dataset.
    
    Args:
    - model_path (str): Path to the trained model file.
    - input_dir (str): Directory with input (hazed) images.
    - target_dir (str): Directory with target (dehazed) images.
    - size (tuple): Desired size for resizing the images.

    Returns:
    - dict: Evaluation metrics (MSE, PSNR, SSIM).
    """
    # Load the trained model with custom objects
    with keras.utils.custom_object_scope({'brelu': brelu, 'Maxout': Maxout}):
        model = keras.models.load_model(model_path)
    
    # Load and preprocess the dataset
    X, y_true = load_dataset(input_dir, target_dir, size)

    # Make predictions
    y_pred = model.predict(X)

    # Compute evaluation metrics
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    psnr = peak_signal_noise_ratio(y_true, y_pred)
    ssim = structural_similarity(y_true, y_pred, multichannel=True)

    return {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim}

if __name__ == '__main__':
    input_dir = 'D:\\image-dehazing\\model\\data\\train\\input'
    target_dir = 'D:\\image-dehazing\\model\\data\\train\\target'
    model_path = 'model\\models\\dehazing_model.h5'  # Path to the trained model

    metrics = evaluate_model(model_path, input_dir, target_dir)
    print("Evaluation metrics:")
    print(f"MSE: {metrics['MSE']}")
    print(f"PSNR: {metrics['PSNR']}")
    print(f"SSIM: {metrics['SSIM']}")
