import cv2
import numpy as np

# Constants
KNOWN_DISTANCE = 50  # 已知距离
KNOWN_WIDTH = 0.5 # 已知宽度0.787
KNOWN_HEIGHT = 0.5  # 已知高度0.787
KNOWN_FOCAL_LENGTH = 1000  # 已知焦距

# Global variables
MaxDistance = float('-inf')  # 最大距离
MinDistance = float('inf')  # 最小距离

def GetPicture(SrcImage, choice):
    global MaxDistance, MinDistance

    # Display input image
    cv2.namedWindow("输入窗口", 0)
    cv2.resizeWindow("输入窗口", 600, 600)
    cv2.imshow("输入窗口", SrcImage)

    # Calculate focal length
    FocalLength = CalculateFocalDistance(SrcImage)

    # Find marker and calculate distance
    Marker = FindMarker(SrcImage)
    DistanceInches = GetTheDistanceToCamera(KNOWN_WIDTH, KNOWN_FOCAL_LENGTH, Marker[1][0])
    DistanceInches *= 2.54  # Convert to cm

    # Update max and min distances
    MaxDistance = max(MaxDistance, DistanceInches)
    MinDistance = min(MinDistance, DistanceInches)

    print(f"DistanceInches(cm): {DistanceInches}")

    # Display distance on image
    cv2.putText(SrcImage, f"distance: {DistanceInches:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10, cv2.LINE_8)
    cv2.namedWindow("输出窗口", 0)
    cv2.resizeWindow("输出窗口", 600, 600)
    cv2.imshow("输出窗口", SrcImage)

    if choice != 1:
        cv2.waitKey(1)
    else:
        cv2.waitKey(0)

def GetCamera(choice):
    global MaxDistance, MinDistance

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        GetPicture(frame, choice)
        if cv2.waitKey(10) == 32:
            break

    print(f"图像中检测过的轮廓中，最大距离为：{MaxDistance}cm")
    print(f"图像中检测过的轮廓中，最小距离为：{MinDistance}cm")
    capture.release()
    cv2.destroyAllWindows()

def GetVideo(choice):
    global MaxDistance, MinDistance

    capture = cv2.VideoCapture("视频的绝对路径.mp4")

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        GetPicture(frame, choice)
        cv2.waitKey(1)

    print(f"图像中最大距离为：{MaxDistance}cm")
    print(f"图像中最小距离为：{MinDistance}cm")
    cv2.waitKey()

def GetTheDistanceToCamera(KnownWidth, FocalLength, PerWidth):
    return (KnownWidth * FocalLength) / PerWidth

def FindMarker(SrcImage):
    GrayImage = cv2.cvtColor(SrcImage, cv2.COLOR_BGR2GRAY)
    GaussImage = cv2.GaussianBlur(GrayImage, (3, 7), 3)
    EdgeImage = cv2.Canny(GaussImage, 100, 200)

    contours, _ = cv2.findContours(EdgeImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(largest_contour)

def CalculateFocalDistance(Image):
    # Placeholder for actual focal length calculation
    return KNOWN_FOCAL_LENGTH

if __name__ == "__main__":
    choice = int(input("请输入你想选择的模式\n识别图片请输入：1\n实时摄像头识别请输入：2\n读取视频请输入：3\n"))

    if choice == 1:
        SrcImage = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
        GetPicture(SrcImage, choice)
        print(f"图像中检测过的轮廓中，最大距离为：{MaxDistance}cm")
        print(f"图像中检测过的轮廓中，最小距离为：{MinDistance}cm")
    elif choice == 2:
        GetCamera(choice)
    elif choice == 3:
        GetVideo(choice)