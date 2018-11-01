import numpy as np 
import cv2 

def order_points(pts):
	#인풋 pts: 사각형의 각 꼭짓점의 (x,y)좌표를 담은 list

	# 4개의 숫자를 담을 numpy array 를 만든다. 
	# 첫번째 숫자: 왼쪽 위, 2번째: 오른쪽 위, 3번째: 왼쪽 밑, 4번째: 오른쪽 밑
	rect = np.zeros((4,2), dtype="float32")

	# 왼쪽 위 숫자가 합이 가장 크고, 오른쪽 밑 숫자가 합이 가장 작게 만든다 
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 오른쪽 위 숫자: 가장 작은 차, 왼쪽 밑: 가장 큰 차 
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect  # tl: top left, br: bottom right. 순서가 정말 중요하다!

	# 새로운 이미지의 가로를 계산한다. 두 모서리 중 긴 것을 사용 
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# 새로운 이미지의 세로를 계산한다. 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	#위에서 내려다 보는 view의 이미지를 만든다. Destination points -> 순서 중요! 
	# top left -> top right -> bottom right -> bottom left의 순서로 (시계방향)
	dst = np.array([[0,0], [maxWidth - 1, 0], 
					[maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

	# cv2.getPerspectiveTransform: 2개 argument 필요. rect -> 오리지널 이미지의 ROI, dst -> 바뀔 포인트의 리스트 
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped
