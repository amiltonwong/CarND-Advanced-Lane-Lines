import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

def thresholded_binary(undistorted_img):
    """
    :param undistorted_img:
        Source image (undistorted) for thresholding (combination thresholding from Sobel x gradient + HLS S channel + HSV V channel)
    
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
    s_binary = np.zeros_like(s_channel)
    s_thresh_min = 100
    s_thresh_max = 255
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1    

    # Convert to HSV color space and separate the V channel
    # Note: img is the undistorted image
    hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HSV).astype("float")
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_thresh_min = 50
    v_thresh_max = 255
    v_binary[(v_channel >= v_thresh_min) & (v_channel <= v_thresh_max)] = 1  
    
    # use bitand to obtian the threshold binary for s channel and v channeel
    s_v_binary = np.zeros_like(s_channel)
    s_v_binary[(s_binary == 1) & (v_binary == 1)] =  1
    
    # Stack each channel to view their individual contributions in green (sxbinary) and blue (s_v_binary) respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_v_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_v_binary == 1) | (sxbinary == 1)] = 1

    # create a mask to filter out pixels inside the shape from src points
    vertices = np.array([[[575+50, 464+50],
                 [730-50, 464+50],
                 [1100-100, 682],
                 [280+100, 682]]], dtype=np.int32 )
    mask = np.ones_like(combined_binary)
    cv2.fillPoly(mask, vertices, 0)
    combined_binary = cv2.bitwise_and(combined_binary, mask)

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