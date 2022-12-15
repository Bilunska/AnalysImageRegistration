import cv2
from matplotlib import pyplot as plt
import time

start = time.time()

# реальний час
img1 = cv2.imread('eye_image_scaled.jpg')
img2 = cv2.imread('eye_realsize.jpg')

# Запускаємо детектор SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Конвертуємо зображення у СІРЕ
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

startDetect = time.time()

# знаходимо ключові точки та дескриптори за допомогою SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

endDetect = time.time()
total_timeDetect = endDetect - startDetect
print("Виявлення та обчислення час (секунди): ", str(total_timeDetect))

startMatching = time.time()

# BFMatcher (Brute-force descriptor matcher) із параметрами за замовчуванням
bf = cv2.BFMatcher(normType = cv2.NORM_L2,
                   crossCheck = True)

matches = bf.match(queryDescriptors = des1,
                          trainDescriptors = des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

endMatching = time.time()
total_timeMatching = endMatching - startMatching
print("Зіставлення час (секунди): ", str(total_timeMatching))

end = time.time()
total_time = end - start
print("Загальний час (секунди): ", str(total_time))

print("Кількість ключових точок: (оригінал) ", len(des1))
print("Кількість ключових точок: (реальний світ) ", len(des2))

print("matches ", len(matches))

# Create images with keypoints
img1=cv2.drawKeypoints(img1, kp1, img1)
cv2.imwrite('sift_keypoints_1.jpg', img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite('sift_keypoints_2.jpg',img2)

result = cv2.drawMatches(img1,kp1,img2,kp2,matches[:300],None,flags = 2)
plt.imshow(result),plt.show()
