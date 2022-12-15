import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

def num_sim(n1, n2):
  """ calculates a similarity score between 2 numbers """
  return print("Rate %:", 1 - abs(n1 - n2) / (n1 + n2))

start = time.time()

img1 = cv.imread('eye.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('eye_in_real.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# число кількості ключових точок
print("Кількість ключових точок: (оригінал) ", len(des1))
print("Кількість ключових точок: (180) ", len(des2))
num_sim(len(des1), len(des2))

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# Apply ratio test
good = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i]=[1,0]
        good.append([m])


print("matches ", len(good))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
