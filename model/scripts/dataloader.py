import os 
import cv2

def load_dataset(input_dir, target_dir, size=(256, 256)):
     """
    Load and preprocess the dataset from input and target directories.
    
    Args:
    - input_dir (str): Directory with input (hazed) images.
    - target_dir (str): Directory with target (dehazed) images.
    - size (tuple): Desired size for resizing the images.

    Returns:
    - np.array: Array of input images.
    - np.array: Array of target images.
    """
    input_images = []
    target_images = []
    for input_image, target_image in zip(os.listdir(input_dir), os.listdir(target_dir)):
        input_image = cv2.imread(os.path.join(input_dir, input_image))
        target_image = cv2.imread(os.path.join(target_dir, target_image))
        input_image = cv2.resize(input_image, size)
        target_image = cv2.resize(target_image, size)
        input_images.append(input_image)
        target_images.append(target_image)
    return np.array(input_images), np.array(target_images)
```