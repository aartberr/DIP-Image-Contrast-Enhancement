#demo
from functions import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#Get the grayscale image
#Set the filepath to the image file
filename = "input_img.png"

#Read the image into a PIL entity
img = Image.open(fp=filename)

#Keep only the Luminance component of the image
bw_img = img.convert("L")

#Obtain the underlying np array
img_array = np.array(bw_img)

#Equalized image
equalized_img=perform_global_hist_equalization(img_array)

# Plot the original image and its histogram
plt.figure(num='Figure 1', figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Original Histogram')
plt.hist(img_array.flatten(), bins=256, range=(0, 255), color='black', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Plot the equalized image and its histogram
plt.subplot(2, 2, 3)
plt.title('Equalized Image')
plt.imshow(equalized_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Equalized Histogram')
plt.hist(equalized_img.flatten(), bins=256, range=(0, 255), color='black', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show(block=False)

# Plot the adapted histogram equalized image without interpolation and its histogram 
interpolation=0
adapted=perform_adaptive_hist_equalization(img_array ,48 ,64,interpolation)
plt.figure(num='Figure 2', figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title('Adapted Histogram Equalized Image without interpolation')
plt.imshow(adapted, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.hist(adapted.flatten(), bins=256, range=(0, 255), color='black', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show(block=False)


# Plot the adapted histogram equalized image with interpolation and its histogram
interpolation=1
adapted_w_interpolation=perform_adaptive_hist_equalization(img_array ,48 ,64,interpolation)
plt.figure(num='Figure 3', figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title('Adapted Histogram Equalized Image with interpolation')
plt.imshow(adapted_w_interpolation, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.hist(adapted_w_interpolation.flatten(), bins=256, range=(0, 255), color='black', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()