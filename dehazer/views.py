from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import os
import cv2
from model.src.tensorflow.DehazeNet import dehazenet
from model.src.tensorflow.preprocessing import preprocess_image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import uuid
# Load models
model_paths = {
    'model1': 'D:\image-dehazing\model\models\dehazing_model.keras',
}

models = {}
def load_models():
    keras.saving.get_custom_objects().clear()

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

    for model_name, model_path in model_paths.items():
        models[model_name] = load_model(model_path)

load_models()

def index(request):
    return render(request, 'index.html')
def dehaze(request):
    if request.method == 'POST':
        file = request.FILES['file']
        model_name = request.POST['model']
        
        if model_name not in models:
            return JsonResponse({'error': 'Invalid model selected'}, status=400)

        # Save uploaded file to a temporary location
        temp_file = default_storage.save('temp.jpg', ContentFile(file.read()))
        temp_file_path = default_storage.path(temp_file)
        
        # Preprocess the image
        input_image = preprocess_image(temp_file_path, size=(144, 144))
        input_image = np.expand_dims(input_image, axis=0)

        # Get the selected model and run prediction
        model = models[model_name]
        dehazed_image = model.predict(input_image)
        dehazed_image = (dehazed_image[0] * 255).astype(np.uint8)

        # Generate a unique file name for the dehazed image
        unique_filename = f'dehazed_{uuid.uuid4().hex}.jpg'
        output_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
        cv2.imwrite(output_path, dehazed_image)

        # Get the URL of the dehazed image
        dehazed_image_url = os.path.join(settings.MEDIA_URL, unique_filename)
        return JsonResponse({'dehazed_image_url': dehazed_image_url})

    return JsonResponse({'error': 'Invalid request'}, status=400)
