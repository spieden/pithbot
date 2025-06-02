import cv2
import numpy as np
import os
from pathlib import Path

def extract_comic_panels(image_path, output_dir, buffer_ratio=0.15, min_panel_area=5000):
    """
    Extract comic panels from an image with buffer space below for captions.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save extracted panels
        buffer_ratio: Ratio of panel height to add as buffer below panel
        min_panel_area: Minimum area (in pixels) for a region to be considered a panel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect components within panels
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate (left to right, top to bottom)
    def sort_key(contour):
        x, y, w, h = cv2.boundingRect(contour)
        # Use row-major order (top-to-bottom, then left-to-right)
        row = y // 100  # Approximate row number
        return (row, x)

    contours = sorted(contours, key=sort_key)

    # Process each contour
    for i, contour in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Skip if area is too small
        if w * h < min_panel_area:
            continue

        # Calculate buffer height for caption
        buffer_height = int(h * buffer_ratio)

        # Ensure buffer doesn't go beyond image bounds
        bottom_y = min(y + h + buffer_height, image.shape[0])

        # Extract panel with buffer
        panel = image[y:bottom_y, x:x+w]

        # Save panel
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_panel_{i+1}.jpg")
        cv2.imwrite(output_path, panel)
        print(f"Saved panel {i+1} to {output_path}")

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python extract_comics.py <input_image_path> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    extract_comic_panels(input_path, output_dir)
