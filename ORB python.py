
import cv2
from matplotlib import pyplot as plt
import time

start = time.time()

img1 = cv2.imread('eye_image_scaled.jpg')
img2 = cv2.imread('eye_realsize.jpg')

# Initiate ORB detector
orb = cv2.ORB_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

startDetect = time.time()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

endDetect = time.time()
total_timeDetect = endDetect - startDetect
print("Виявлення та обчислення час (секунди): ", str(total_timeDetect))

startMatching = time.time()

# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the test image
matches = bf.match(queryDescriptors = des1,
                          trainDescriptors = des2)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)

endMatching = time.time()
total_timeMatching = endMatching - startMatching
print("Зіставлення час (секунди): ", str(total_timeMatching))

end = time.time()
total_time = end - start
print("Загальний час (секунди): ", str(total_time))

print("Кількість ключових точок: (оригінал) ", len(des1))
print("Кількість ключових точок: (реальний світ) ", len(des2))

# Print total number of matching points between the training and query images
print("matches ", len(matches))

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite('orb_keypoints_1.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite('orb_keypoints_2.jpg',img2)

result = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags = 2)
plt.imshow(result),plt.show()
