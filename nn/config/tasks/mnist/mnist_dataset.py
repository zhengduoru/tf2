import tensorflow as tf
from datasets.dataset import Dataset
import os

class Dataset(Dataset):

    def input_fn(self, mode):
        # Load and prepare the MNIST dataset.
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        # Add a channels dimension
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
        
        
        #Use tf.data to batch and shuffle the dataset:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(32)
        
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        if mode == 'train': 
            return train_ds
        if mode == 'eval': 
            return test_ds
