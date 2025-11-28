# file_utils.py
"""This module contains utility functions for file operations."""
import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split

def search_files(directory, pattern):
    """Searches for files based on a pattern.

    Args:
        directory (str): The directory to search in.
        pattern (str): The glob pattern to match.

    Returns:
        list: A list of matching file paths.
    """
    return glob.glob(os.path.join(directory, pattern), recursive=True)

def copy_move_file(src, dst, move=False):
    """Copies or moves a file.

    Args:
        src (str): The source file path.
        dst (str): The destination file path.
        move (bool): If True, moves the file. Otherwise, copies it.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy(src, dst)

def create_directories(paths):
    """Creates directories.

    Args:
        paths (list): A list of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def split_dataset(image_paths, label_paths, train_ratio=0.8, random_state=42):
    """Splits a dataset into training and validation sets.

    Args:
        image_paths (list): List of paths to the images.
        label_paths (list): List of paths to the labels.
        train_ratio (float): The ratio of the training set.
        random_state (int): The random seed for shuffling.

    Returns:
        tuple: (train_images, val_images, train_labels, val_labels)
    """
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, train_size=train_ratio, random_state=random_state
    )
    return train_images, val_images, train_labels, val_labels
