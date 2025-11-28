"""
This is the main entry point for the ROP AI Project CLI.
"""
import argparse
import os
from ultralytics import YOLO
from src.utils import yaml_utils, file_utils, image_processing

def prepare_data(config_path):
    """Prepares data for training."""
    config = yaml_utils.read_yaml(config_path)
    print("Loaded config:", config)

    # 1. Create directories
    print("Creating directories...")
    base_dir = config['base_dir']
    raw_images_dir = os.path.join(base_dir, 'raw_images')
    processed_images_dir = os.path.join(base_dir, 'processed_images')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    file_utils.create_directories([raw_images_dir, processed_images_dir, train_dir, val_dir])

    # 2. Extract frames from video
    print("Extracting frames from video...")
    video_path = config['video_path']
    image_processing.extract_frames_from_video(video_path, raw_images_dir)

    # 3. Process each image (detect, crop, fill, resize)
    print("Processing images...")
    raw_image_files = file_utils.search_files(raw_images_dir, '**/*.jpg')
    for img_path in raw_image_files:
        # Simple processing pipeline
        processed_path = os.path.join(processed_images_dir, os.path.basename(img_path))
        image_processing.resize_image(img_path, processed_path, tuple(config['image_size']))

    # 4. Split dataset
    print("Splitting dataset...")
    processed_image_files = file_utils.search_files(processed_images_dir, '**/*.jpg')
    # Assuming labels have the same name but with .txt extension
    label_files = [f.replace('.jpg', '.txt').replace('processed_images', 'labels') for f in processed_image_files]

    train_images, val_images, train_labels, val_labels = file_utils.split_dataset(
        processed_image_files, label_files, train_ratio=config['train_ratio']
    )

    # 5. Copy files to train/val directories
    print("Copying files to train/val directories...")
    for img, lbl in zip(train_images, train_labels):
        file_utils.copy_move_file(img, os.path.join(train_dir, 'images', os.path.basename(img)))
        if os.path.exists(lbl):
            file_utils.copy_move_file(lbl, os.path.join(train_dir, 'labels', os.path.basename(lbl)))

    for img, lbl in zip(val_images, val_labels):
        file_utils.copy_move_file(img, os.path.join(val_dir, 'images', os.path.basename(img)))
        if os.path.exists(lbl):
            file_utils.copy_move_file(lbl, os.path.join(val_dir, 'labels', os.path.basename(lbl)))

    # 6. Generate dataset YAML
    print("Generating dataset YAML...")
    dataset_yaml_path = os.path.join(base_dir, 'dataset.yaml')
    yaml_utils.generate_dataset_yaml(
        dataset_yaml_path,
        os.path.join(train_dir, 'images'),
        os.path.join(val_dir, 'images'),
        config['class_names']
    )
    print(f"Data preparation complete. Dataset YAML created at: {dataset_yaml_path}")

def train(model_path, data_yaml_path, epochs):
    """Trains a YOLO model."""
    model = YOLO(model_path)
    model.train(data=data_yaml_path, epochs=epochs)
    print(f"Training complete. Model saved to runs/detect/train/")

def inference(model_path, source_path):
    """Runs inference with a YOLO model."""
    model = YOLO(model_path)
    model(source_path, save=True)
    print(f"Inference complete. Results saved to runs/detect/predict/")

def main():
    parser = argparse.ArgumentParser(description='ROP AI Project CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # prepare_data sub-command
    parser_prepare_data = subparsers.add_parser('prepare_data', help='Prepare data for training')
    parser_prepare_data.add_argument('--config', type=str, required=True, help='Path to the data preparation config file')

    # train sub-command
    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('--model', type=str, required=True, help='Path to the model file (e.g., yolov8n.pt)')
    parser_train.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file')
    parser_train.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    # inference sub-command
    parser_inference = subparsers.add_parser('inference', help='Run inference with a model')
    parser_inference.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser_inference.add_argument('--source', type=str, required=True, help='Path to the image or directory for inference')

    args = parser.parse_args()

    if args.command == 'prepare_data':
        prepare_data(args.config)
    elif args.command == 'train':
        train(args.model, args.data, args.epochs)
    elif args.command == 'inference':
        inference(args.model, args.source)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()