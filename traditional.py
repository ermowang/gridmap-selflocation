import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def SGBM(imgL,imgR):
    window_size = 8
    stereo = cv2.StereoSGBM_create(
                minDisparity = 1,
                numDisparities = 128, 
                blockSize = 8,
                P1 = 8 * 3 * window_size ** 2,  #相邻像素视察变化的制约
                P2 = 32 * 3 * window_size ** 2,   #视差平滑度
                disp12MaxDiff = -1,
                preFilterCap = 1,
                uniquenessRatio = 10,
                speckleWindowSize = 100,
                speckleRange = 100,
                mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(img1,img2).astype(np.float32)/16.0
    return disparity

def matching(imgL,imgR):
    K = np.mat([[9.0216789316569793e+02, 0., 6.5417493255108036e+02],
                [0.,9.1544365411265187e+02, 2.8505401947910565e+02],
                [0,0,1]])
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    kp1,desc1 = detector.detectAndCompute(imgL,None)
    kp2,desc2 = detector.detectAndCompute(imgR,None)

    raw_matches = matcher.knnMatch(desc1,desc2,2)
    good = []

    for m,n in raw_matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        #求基础矩阵和本质矩阵以及外参
        F,F_mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC,5.0)   
        E,mask = cv2.findEssentialMat(src_pts,dst_pts,K,method=cv2.RANSAC,prob=0.999,threshold=1)   
        retval,R,T,mask = cv2.recoverPose(E,src_pts,dst_pts,K )
        
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(good),10))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = (0,0,255),
                       matchesMask = matchesMask,
                       flags = 2)# draw only inliers

    match = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return match

img1 = cv2.imread("D:\\1\\000324.png")
img2 = cv2.imread("D:\\2\\000325.png")
#img1 = cv2.imread("D:\\depth.jpg")
#img2 = cv2.imread("D:\\000000_10.png")
time_start=time.time()
depthmap = SGBM(img1,img2)
time_end=time.time()
print('totally cost',time_end-time_start)
matchmap = matching(img1,img2)
plt.imshow(depthmap,'gray')
plt.show()
plt.imshow(matchmap,'gray')
plt.show()
cv2.imwrite("D:\\depth.jpg", depthmap)
cv2.imwrite("D:\\match.jpg", matchmap)
cv2.waitKey()
cv2.destroyAllWindows()
