# yaml_utils.py
"""This module contains utility functions for handling YAML files."""
import yaml

def read_yaml(file_path):
    """Reads a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The content of the YAML file.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(data, file_path):
    """Writes data to a YAML file.

    Args:
        data (dict): The data to write.
        file_path (str): The path to the YAML file.
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def generate_dataset_yaml(output_path, train_dir, val_dir, class_names):
    """Generates a YAML file for a dataset.

    Args:
        output_path (str): The path to save the YAML file.
        train_dir (str): The path to the training data.
        val_dir (str): The path to the validation data.
        class_names (list): A list of class names.
    """
    dataset_yaml = {
        'train': train_dir,
        'val': val_dir,
        'nc': len(class_names),
        'names': class_names
    }
    write_yaml(dataset_yaml, output_path)
