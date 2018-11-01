from transform import four_point_transform 
import numpy as np 
import argparse
import cv2 
import imutils
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

# Step 1: 모서리 찾기 
# 원본사진의 높이와, 바뀐 사진의 높이의 비율 계산하기 
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# 사진을 검은색으로 바꾸고, blur 하고 -> 이미지에서 모서리 찾기 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

# Step 2: edged 이미지에서 윤곽을 찾아낸다 
# 가장 큰것만 남기고, screen contour 를 사용한다 
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts: 
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break
# # show the contour (outline) of the piece of paper
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 3: Transform & Threshold 
warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

#종이의 흑/백 효과를 주기 위해서 grayscale 로 바꾸고 threshold 를 만든다 
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255 
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
