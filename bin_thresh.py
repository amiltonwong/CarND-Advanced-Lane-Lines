import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

def thresholded_binary(undistorted_img):
    """
    :param undistorted_img:
        Source image for thresholding that has already been undistorted.
    
    :return:
        Returns a colour binary image for visualisation purposes and a
        binary thresholded image for use in lane finding.
    """
    
    # Sobel x gradient
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS).astype("float")
    s_channel = hls[:,:,2]
        
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green (sxbinary) and blue (s_binary) respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary

# load undistorted image
exampleImg_undistort  = cv2.imread('./test_images/test1_undist.jpg')
exampleImg_undistort = cv2.cvtColor(exampleImg_undistort, cv2.COLOR_BGR2RGB)

color_binary, combined_binary = thresholded_binary(exampleImg_undistort)

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title("Stacked thresholds")
ax1.imshow(color_binary)
ax1.axis("off");

# save as file
binary = 255 * color_binary.astype("uint8")
binary = cv2.cvtColor(binary, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_images/color_binary_test1.jpg", binary)

ax2.set_title("Combined S channel and gradient thresholds")
ax2.imshow(combined_binary, cmap="gray")
ax2.axis("off");
plt.show()

# save as file
binary = 255 * combined_binary.astype("uint8")
cv2.imwrite("output_images/combined_binary_test1.jpg", binary)