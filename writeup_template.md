**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image1.5]: ./examples/corner_found.jpg "Corners found"
[image2]: ./figure_1.png "Road image and undistortion"
[image3]: ./output_images/combined_binary_test1.jpg "Binary thresholded output"
[image4]: ./output_images/unwarped.png "Warped binary thresholded output"
[image4.5]: ./output_images/hist1.png "Histogram"
[image5]: ./output_images/detect_lanes_test1.png "Detect lanes"
[image6]: ./output_images/result_test1.jpg "Final Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

**1. Camera Calibration - Compute camera calibration using chessboard images.**

The code for this step is contained in the camera_cal/cam_cal.py.
```
$ cd camera_cal
$ python cam_cal.py
```
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The calibration is performed using cv2 function, mainly on: cv2.findChessboardCorners() and cv2.calibrateCamera()
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients: 

![alt text][image1]

![alt text][image1.5]



**2. Pipeline (single images) - Apply distortion correction on road images (located at test_images/)**

The code for this step is contained in undistort_img.py
```
$ python undistort_img.py
```

The result of distortion correction on road images (test1.jpg) is shown :
![alt text][image2]

The above figure is an example of an original and undistorted image. (e.g. test1.jpg). Note that the position of white car in undistorted image is further back after correcting distortion. See line13-25 in ```undistort_img.py```, which mainly applies cv2.undistort() after loading camera matrix and distortion coefficients. 

**3. Create a thresholded binary image using color transforms, gradient**

The code for this step is contained in bin_thresh.py
```
$ python bin_thresh.py
```

To generate a binary image, I mainly use a combination of color and gradient threshold approach. Color gradient x thresholding steps are listed at line (17-27) in ```bin_thresh.py```, while s channel from HSV color transform thresholding are listed at line (29-38).
Firstly, I use the OpenCV Sobel function cv2.sobel() to obtain the gradient in X direction (finding vertical lines is better) and threshold out the range of [20,100].
Secondly, I convert the image to HSV colour-space and then threshold the S-channel in range of [170, 255].
Finally, I combine these two operations using bitwise-or operation to obtain the output binary thresholded image (combined_binary.jpg).

The following figure is the binary thresholded output (sobel x gradient + HLS s channel threshold) of test image (test1.jpg)

![alt text][image3]

**4. Using perspective transformation to rectify binary image ("birds-eye view")**

The code for this step is contained in the perspective_transform.py, which includes a function called `unwarp()`, which appears in lines 5 through 12 in the file `perspective_transform.py`.  The `unwarp()` function takes a binary undistorted image as input  (`img`), as well as source (`src`) and destination (`dst`) points. For the actual perspective transform,  I use the OpenCV `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to execute. Then, I chose the hardcode the source and destination points in the following manner:
```
src = np.float32([(575,464),
                  (730,464), 
                  (280,682), 
                  (1100,682)])
dst = np.float32([(200,0),
                  (w-200,0),
                  (200,h),
                  (w-450,h)])
```

![alt text][image4]

The top-down or birds-eye-view of the road of the rectify binary image ("birds-eye view") for test1.jpg is shown as the above figure.
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

**5. Detect lane pixels and fit to find the lane boundary**

The code for this step is contained in `detect_lane_curvature.py`

I first take a histogram along all the columns in the lower half of the image. The corresponding part locates in line 10. The histogram result looks like the following figure. We can observe two peaks locating at x~=220 and ~=1100.

![alt text][image4.5]

Then, based on this histogram, I choose these two most prominent 
peaks in histogram as good indicators of the x-position of the base 
of the lane lines. I use that as a starting point for where to search for the lines. 
From that point, I use a sliding window, placed around the line centers, 
to find and follow the lines up to the top of the frame(use 9 windows from the bottom to the top of the image). The corresponding part of code
located in line 19-82 in detect_lane_curvature.py. Then, I use a second order polynomial to fit 
the detected left and right lanes (line 85-86).

To visualize the result, I generate x and y values for plotting and we can view the result from the following figure:

![alt text][image5]

We could see the detected curves fit well with the orginal white lanes.  We've estimated which pixels belong to the left and right lane lines (shown in red and blue, respectively in the figure above), and we've fit a polynomial to those pixel positions (the yellow curves inside the red and blue strip).

**6. Determine the curvature of the lane and vehicle position with respect to center**

The code for this step is contained in `detect_lane_curvature_unwarp.py`
The corresponding part for curvature computation is listed in line 110-113 in `detect_lane_curvature_unwarp.py` . I use this formula (http://www.intmath.com/applications-differentiation/8-radius-curvature.php) to compute. Then, we compute the x intercept for left lane and right lane at the bottom of the image and compute the center between them. We assume the camera is located at the center of the front view. Thus deviation of vehicle position with respect to the center of the lane is computed as the difference between center of lane and the half of scene width (line 126-133). 

**7. Warp the detected lane boundaries back onto the original image**

code: line 138-192 in `final.py`
We reuse Minv matrix to unwarp the detected lane back to original image, which is listed in draw_lanes_on_image() in line 138-175 in `final.py`. cv2.addWeighted() is used to combine the result with the original image, also the radius of curvature of left and right lanes, deviation position from vehicle are overlaid on it. The final result is shown as the following figure. We could view the result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]


---

**8. The pipeline for video (project_video.mp4) **

The following link shows the final video output for "project_video.mp4".  My pipeline perform reasonably well on the entire project video .

Here's the [link](https://youtu.be/mBHRAK3qlGI)
---

**9. Discussion**

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

