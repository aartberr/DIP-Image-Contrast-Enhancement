# Histogram Equalization for Image Enhancement

This project was created as part of a course assignment for the **DIP (Digital Image Processing)** class at **ECE, AUTH University**.

This project applies **histogram equalization** to enhance grayscale images by increasing contrast. The process involves distributing pixel intensity values more evenly across the histogram, improving visibility in regions with low contrast. The implementation includes both **global** and **local** histogram equalization, with an interpolation-based refinement for better results.

## Features
- **Global Histogram Equalization**: Enhances contrast across the entire image.
- **Local Histogram Equalization**: Adjusts contrast in smaller regions.
- **Interpolation-based Enhancement**: Further improves local equalization results.

## Usage 
1. Place your grayscale image named input_img.png in the same folder as the script.

2. Run the demo.py script:
```bash
python demo.py
```

3. This will output the following:

- **Global Histogram Equalization**: An image with improved contrast using the global histogram equalization method.
- **Local Histogram Equalization**: An image with improved contrast using local histogram equalization (both with and without interpolation).
  
In addition to the enhanced images, the histograms will be displayed for both the original and equalized images.

4. The enhanced images and histograms will be shown as plots.

