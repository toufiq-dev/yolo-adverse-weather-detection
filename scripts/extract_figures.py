#!/usr/bin/env python3
"""
Script to extract figures from thesis PDF
用于从论文PDF中提取图片的脚本
"""

import os
import sys
import fitz  # pymupdf


def extract_figures_from_pdf(pdf_path, output_dir, page_range=None):
    """
    Extract all figures from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save extracted images
        page_range (tuple): Optional (start, end) page numbers (0-indexed)
    
    Returns:
        int: Number of images extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Determine page range
    if page_range:
        start, end = page_range
        pages = range(start, min(end, len(doc)))
    else:
        pages = range(len(doc))
    
    image_count = 0
    
    for page_num in pages:
        page = doc[page_num]
        images = page.get_images()
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_count += 1
            image_filename = os.path.join(
                output_dir, 
                f"figure_page{page_num+1}_{img_idx+1}.{image_ext}"
            )
            
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            
            print(f"Extracted: {image_filename}")
    
    doc.close()
    return image_count


def main():
    """Main function"""
    # Default paths
    default_pdf = "docs/thesis-original/1910046_Robiul_correction.pdf"
    default_output = "docs/figures"
    
    # Parse command line arguments
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Page range for figures (pages 56-78 based on thesis structure)
    page_range = (55, 78)  # 0-indexed
    
    print(f"Extracting figures from: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Page range: {page_range[0]+1}-{page_range[1]+1}")
    print("-" * 50)
    
    # Extract figures
    image_count = extract_figures_from_pdf(pdf_path, output_dir, page_range)
    
    print("-" * 50)
    print(f"Total images extracted: {image_count}")
    print("Done!")


if __name__ == "__main__":
    main()
