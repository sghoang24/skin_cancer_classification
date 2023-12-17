"""
Some helper functions for TensorFlow2.0, including:
    - get_dataset(): download dataset from TensorFlow.
    - get_mean_and_std(): calculate the mean and std value of dataset.
    - normalize(): normalize dataset with the mean the std.
    - dataset_generator(): return `Dataset`.
    - progress_bar(): progress bar mimic xlua.progress.
"""
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import concurrent.futures
import multiprocessing

from PIL import Image
import subprocess as sp

import tensorflow as tf
from tensorflow.keras import datasets
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # used for converting labels to one-hot-encoding

from sklearn.model_selection import train_test_split

# Create an ImageDataGenerator object with the desired transformations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

padding = 4
image_size = (75, 100)
target_size = (image_size[0] + padding*2, image_size[1] + padding*2)
max_images_per_class = 2500

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def mask_unused_gpus(leave_unmasked=1):
  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) < leave_unmasked: ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
  except Exception as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

def read_data():
    """Read data"""
    train_dir = '/kaggle/input/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
    test_dir = '/kaggle/input/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

    # Create dataframes
    train_df = pd.DataFrame(columns=['image_path', 'label'])
    test_df = pd.DataFrame(columns=['image_path', 'label'])

    # Add images paths and labels to dataframes
    for label, directory in enumerate(os.listdir(train_dir)):
        for filename in os.listdir(os.path.join(train_dir, directory)):
            image_path = os.path.join(train_dir, directory, filename)
            train_df = train_df.append({'image_path': image_path, 'label': label}, ignore_index=True)

    for label, directory in enumerate(os.listdir(test_dir)):
        for filename in os.listdir(os.path.join(test_dir, directory)):
            image_path = os.path.join(test_dir, directory, filename)
            test_df = test_df.append({'image_path': image_path, 'label': label}, ignore_index=True)
            
    # Combine train_df and test_df into one dataframe
    df = pd.concat([train_df, test_df], ignore_index=True)
    del test_df, train_df
    return df

def loading_and_resize(df):
    """Loading and resize image"""
    # Group by label column and take first max_images_per_class rows for each group
    df = df.groupby("label").apply(lambda x: x.head(max_images_per_class)).reset_index(drop=True)
    return df

# Define a function to resize image arrays
def resize_image_array(image_path):
    """Resize image arrays"""
    return np.asarray(Image.open(image_path).resize(image_size[::-1]))

def resize_image(df):
    # Get the number of CPU cores available
    max_workers = multiprocessing.cpu_count()
    print("Max worker:", max_workers)

    # Use concurrent.futures to parallelize the resizing process
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to apply the function to each image path in the DataFrame
        image_arrays = list(executor.map(resize_image_array, df['image_path'].tolist()))

    # Add the resized image arrays to the DataFrame
    df['image'] = image_arrays
    del image_arrays
    return df

def prepare_dataset(df, num_classes=9):
    mask_unused_gpus()
    # Allow gpu usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPUs:", gpus)
    try:
        tf.config.experimental.set_memory_growth = True
    except Exception as ex:
        print(ex)

    """Prepare, parse and process a dataset to unit scale and one-hot labels."""
    features = df.drop(columns=['label', 'image_path'], axis=1)
    target = df['label']

    # Train & test split
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)
    print("split train & test")

    x_train = np.asarray(x_train['image'].tolist())
    x_test = np.asarray(x_test['image'].tolist())

    # Normalize images
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_test_mean) / x_test_std

    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    print("one hot y_train, y_test")

    # Train & validate split
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
    print("split train & validation")

    # Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
    x_train = x_train.reshape(x_train.shape[0], *(image_size + (3,)))
    x_test = x_test.reshape(x_test.shape[0], *(image_size + (3,)))
    x_validate = x_validate.reshape(x_validate.shape[0], *(image_size + (3,)))

    y_train = y_train.astype(int)
    y_validate = y_validate.astype(int)

    return x_train, x_validate, x_test, y_train, y_validate, y_test

def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std

def data_augmentation(df):
    # Create an empty dataframe to store the augmented images
    augmented_df = pd.DataFrame(columns=['image_path', 'label', 'image'])

    # Loop through each class label and generate additional images if needed
    for class_label in df['label'].unique():
        # Get the image arrays for the current class
        image_arrays = df.loc[df['label'] == class_label, 'image'].values
        
        # Calculate the number of additional images needed for the current class
        num_images_needed = max_images_per_class - len(image_arrays)
        
        # Generate augmented images for the current class
        if num_images_needed > 0:
            # Select a random subset of the original images
            selected_images = np.random.choice(image_arrays, size=num_images_needed)
            
            # Apply transformations to the selected images and add them to the augmented dataframe
            for image_array in selected_images:
                # Reshape the image array to a 4D tensor with a batch size of 1
                image_tensor = np.expand_dims(image_array, axis=0)
                
                # Generate the augmented images
                augmented_images = datagen.flow(image_tensor, batch_size=1)
                
                # Extract the augmented image arrays and add them to the augmented dataframe
                for i in range(augmented_images.n):
                    augmented_image_array = augmented_images.next()[0].astype('uint8')
                    augmented_df = augmented_df.append({'image_path': None, 'label': class_label, 'image': augmented_image_array}, ignore_index=True)
        
        # Add the original images for the current class to the augmented dataframe
        original_images_df = df.loc[df['label'] == class_label, ['image_path', 'label', 'image']]
        augmented_df = augmented_df.append(original_images_df, ignore_index=True)

    # Group the augmented dataframe by the 'label' column and filter out extra images
    df = augmented_df.groupby('label').head(max_images_per_class)
    
    del augmented_df

    # Use the augmented dataframe for further processing
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)

# def _augment_fn(images, labels):
#     images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
#     images = tf.image.random_crop(images, (image_size, image_size, 3))
#     images = tf.image.random_flip_left_right(images)
#     return images, labels
