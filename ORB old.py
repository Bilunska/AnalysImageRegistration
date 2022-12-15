import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def num_sim(n1, n2):
  """ calculates a similarity score between 2 numbers """
  return print("Rate %:", 1 - abs(n1 - n2) / (n1 + n2))

start = time.time()

# # Read image
# img1 = cv2.imread('eye.jpg')
# img2 = cv2.imread('eye_rotate.jpg')

img1 = cv2.imread('eye.jpg')
img2 = cv2.imread('eye_realsize.jpg')

# Initiate ORB detector
orb = cv2.ORB_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# число кількості ключових точок
print("Кількість ключових точок: (оригінал) ", len(des1))
print("Кількість ключових точок: (180) ", len(des2))
num_sim(len(des1), len(des2))

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite('orb_keypoints_1.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite('orb_keypoints_2.jpg',img2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print("matches ", len(good))

end = time.time()
total_time = end - start
print("Загальний час (секунди): ", str(total_time))

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()
