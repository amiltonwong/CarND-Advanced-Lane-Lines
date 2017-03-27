import numpy as np
import cv2
import matplotlib.pyplot as plt

def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# load undistorted image
#exampleImg_undistort  = cv2.imread('./test_images/test1_undist.jpg')
#exampleImg_undistort = cv2.cvtColor(exampleImg_undistort, cv2.COLOR_BGR2RGB)

# load undistorted binary image
exampleImg_undistort  = cv2.imread('./output_images/combined_binary_test1.jpg')

h,w = exampleImg_undistort.shape[:2]

# define source and destination points for transform
src = np.float32([(575,464),
                  (730,464), 
                  (280,682), 
                  (1100,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)

# Visualize unwarp
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(exampleImg_undistort)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
ax1.set_ylim([h,0])
ax1.set_xlim([0,w])
#ax1.set_title('Undistorted Image', fontsize=30)
ax1.set_title('Undistorted Binary Image', fontsize=30)
ax2.imshow(exampleImg_unwarp)
ax2.set_title('Unwarped Image', fontsize=30)
plt.show()
print('...')

# save as file
# binary = 255 * combined_binary.astype("uint8")
cv2.imwrite("output_images/unwarped_binary_test1.jpg", exampleImg_unwarp)