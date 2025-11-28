# image_processing.py
"""This module contains utility functions for image processing."""
import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """Extracts frames from a video file.

    Args:
        video_path (str): The path to the video file.
        output_dir (str): The directory to save the frames.
        frame_interval (int): The interval at which to extract frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()

def detect_lens_region(image_path):
    """Detects the lens region in an image. (Placeholder)"""
    print(f"Detecting lens region for {image_path}")
    # Placeholder: returns the whole image dimensions
    img = cv2.imread(image_path)
    return 0, 0, img.shape[1], img.shape[0]

def crop_lens(image_path, x, y, w, h, output_path):
    """Crops the lens region from an image."""
    img = cv2.imread(image_path)
    cropped_img = img[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_img)

def fill_background(image_path, output_path):
    """Fills the background of an image. (Placeholder)"""
    print(f"Filling background for {image_path}")
    # Placeholder: just copies the image
    shutil.copy(image_path, output_path)

def resize_image(image_path, output_path, size):
    """Resizes an image."""
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, size)
    cv2.imwrite(output_path, resized_img)

def augment_image(image_path, output_dir):
    """Applies image augmentation. (Placeholder)"""
    print(f"Augmenting {image_path} and saving to {output_dir}")
    # Placeholder: just copies the image
    shutil.copy(image_path, os.path.join(output_dir, os.path.basename(image_path)))
