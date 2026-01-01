# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROP (Retinopathy of Prematurity) AI detection and segmentation project using YOLO and RT-DETR models. The project includes CLI tools for data preparation, model training, and inference on fundus images/videos.

## Development Commands

### Virtual Environment Setup
```bash
# Windows - Activate virtual environment
.\ropenv\Scripts\activate

# Install dependencies
pip install -r ROP_project/requirements.txt
```

### CLI Usage (main.py)

#### Data Preparation
```bash
# Prepare data from config file
python ROP_project/main.py prepare_data --config data_prep_config.yaml
```

#### Model Training
```bash
# Train YOLO/RT-DETR model
python ROP_project/main.py train --model models/yolov8n.pt --data data/dataset.yaml --epochs 50
```

#### Inference
```bash
# Run inference on images/videos
python ROP_project/main.py inference --model runs/detect/train/weights/best.pt --source data/val/images
```

### Jupyter Notebook Workflows
- Model training notebooks in `ROP_project/notebooks/`:
  - `train_yolo-seg.ipynb` - YOLO segmentation training
  - `train_rt-detr-detect.ipynb` - RT-DETR detection training
  - `train_rf-detr-seg.ipynb` - RF-DETR segmentation training

### Best Image Validation
```bash
# Generate best images list
python ROP_project/bestimage_validation/generate_best_images_list.py

# Infer case video
python ROP_project/bestimage_validation/infer_case_video.py
```

## Architecture Overview

### Core Modules
- **main.py**: CLI entry point with three commands (prepare_data, train, inference)
- **src/utils/**: Utility modules for YAML handling, file operations, and image processing
- **notebooks/**: Experimental Jupyter notebooks for model training and evaluation

### Data Organization
- **data/ROP_video/**: Input video files for processing
- **data/ROP_image/**: Extracted and labeled image datasets
- **data/train/**: Training dataset with images/labels in train/valid splits
- **models/**: Pre-trained model weights (.pt files)
- **outputs/**: Inference results and processed videos

### Model Types
- **YOLO11 variants**: yolo11n-seg.pt, yolo11m-seg.pt for segmentation
- **RT-DETR**: rtdetr-l.pt for object detection
- Custom trained models for lens/fundus/disc/macula detection

### Dataset Format
- YOLO format annotations (class_id, x_center, y_center, width, height)
- Segmentation masks in polygon format
- Train/validation split with stratified sampling for balanced datasets

## Key Workflows

### Training Pipeline
1. Data preparation: Extract frames from videos, apply circular masking for lens images
2. Dataset organization: Copy images/labels to train/valid directories
3. Label adjustment: Convert 1-indexed to 0-indexed labels if needed
4. Create data.yaml with class definitions
5. Train model with appropriate hyperparameters (batch size, epochs, augmentations)

### Best Image Selection
- Validates images based on lens detection and retina visibility ratio
- Uses adaptive thresholding to select top quality frames
- Generates Excel reports with selected best images per case

## Important Notes
- Always activate virtual environment before running commands
- Check GPU availability with torch.cuda.is_available() before training
- Use patience parameter to prevent overfitting
- For segmentation tasks, ensure empty label files exist for images without annotations