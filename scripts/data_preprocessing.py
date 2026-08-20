#!/usr/bin/env python3
"""
Data preprocessing utilities for DAWN dataset
用于DAWN数据集的数据预处理工具
"""

import os
import cv2
import numpy as np
from pathlib import Path


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input image
        clip_limit: Clip limit for CLAHE
        grid_size: Grid size for CLAHE
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def apply_dehazing(image):
    """
    Apply dehazing to remove fog effects
    
    Args:
        image: Input image
    
    Returns:
        Dehazed image
    """
    # Simple dehazing using dark channel prior
    # This is a basic implementation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR
    dehazed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return dehazed


def apply_denoising(image, h=10):
    """
    Apply denoising using Fast Non-Local Means
    
    Args:
        image: Input image
        h: Filter strength
    
    Returns:
        Denoised image
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return denoised


def apply_sharpening(image, kernel_size=3):
    """
    Apply sharpening using kernel filter
    
    Args:
        image: Input image
        kernel_size: Size of the sharpening kernel
    
    Returns:
        Sharpened image
    """
    # Create sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    # Apply kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def preprocess_image(image_path, output_path=None):
    """
    Apply full preprocessing pipeline to an image
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image (optional)
    
    Returns:
        Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize to standard size
    image = cv2.resize(image, (640, 640))
    
    # Apply preprocessing steps
    image = apply_clahe(image)
    image = apply_dehazing(image)
    image = apply_denoising(image)
    image = apply_sharpening(image)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
    
    return image


def main():
    """Main function for testing"""
    # Example usage
    input_dir = "data/dawn_dataset/images"
    output_dir = "data/dawn_dataset/processed"
    
    if os.path.exists(input_dir):
        for img_file in os.listdir(input_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file)
                
                try:
                    preprocess_image(input_path, output_path)
                    print(f"Processed: {img_file}")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    else:
        print(f"Input directory not found: {input_dir}")
        print("Please download the DAWN dataset first.")


if __name__ == "__main__":
    main()
