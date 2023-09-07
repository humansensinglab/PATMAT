import cv2
import numpy as np

#read images : src image will be cloned into dst
masked = cv2.imread("./2.jpg") * -1 + 1
gout= cv2.imread("./im2.jpg")
print(masked)
gt = cv2.imread("./data/sam/2.jpg")
#print(gout)
# Create an all white mask
#print(gt.shape)
#print(gt.dtype)
mask = 255 * np.ones(gt.shape, gt.dtype)
obj = (masked * gout).astype('uint8')
im = gt
#print(im.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (int(height/2), int(width/2))

#print(obj.shape)
#print(im.shape)
#print(mask.shape)
# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

# Write results

cv2.imwrite("./cv1.jpg", normal_clone)
cv2.imwrite("./cv2.jpg", mixed_clone)
