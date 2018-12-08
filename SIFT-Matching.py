
# coding: utf-8

# In[7]:

import cv2
import numpy as np
from matplotlib import pyplot as plt

img01 = cv2.imread('./StereoImages/01.jpeg',0)  #queryimage # left image
img02 = cv2.imread('./StereoImages/02.jpeg',0) #trainimage # right image

mtx = np.array([[3.27894981e+03, 0.00000000e+00, 2.05358222e+03],
 [0.00000000e+00, 3.29320263e+03, 1.49021929e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 2.83092458e-01, -1.66799155e+00, -1.62644698e-03,  4.04708463e-04, 3.04420786e+00]])

h, w = img01.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
img1 = cv2.undistort(img01, mtx, dist, None, mtx)
img2 = cv2.undistort(img02, mtx, dist, None, mtx)

# crop the image
# x,y,w,h = roi
# img1 = img1[y:y+h, x:x+w]
# img2 = img2[y:y+h, x:x+w]
# cv2.imwrite('undistorted1.png',img1)
# cv2.imwrite('undistorted2.png',img2)


# In[8]:

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)


# In[9]:

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


# In[4]:

# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# plt.figure(figsize=(15,15))
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()


# In[5]:

# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in xrange(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.3*n.distance:
#         matchesMask[i]=[1,0]
        
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.figure(figsize=(15,15))
# plt.imshow(img3,),plt.show()


# In[10]:

# Normalize for Esential Matrix calaculation
# pts_l_norm = cv2.undistort(np.expand_dims(pts1, axis=1), cameraMatrix=mtx, distCoeffs=None)
# pts_r_norm = cv2.undistort(np.expand_dims(pts2, axis=1), cameraMatrix=mtx, distCoeffs=None)

pts_l = pts1; pts_r = pts2
pts_l_norm = pts1; pts_r_norm = pts2
K_l = mtx
K_r = mtx

E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, mtx)

R1,R2,t = cv2.decomposeEssentialMat(E)
#print R1
#print R2, t
M_r1 = np.hstack((R2, t))
M_r2 = np.hstack((R1, t))
M_r3 = np.hstack((R2, t*(-1)))
M_r4 = np.hstack((R1, t*(-1)))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
P_l = np.dot(K_l,  M_l)
P_r = np.dot(K_r,  M_r4)

print 'F = :', F
print 'E = :', E
print 'R = :', R2
print 't = :', t*(-1)


# point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))

pts4d = cv2.triangulatePoints(P_l, P_r, np.array(np.transpose(pts1)), np.array(np.transpose(pts2)))
X_vector = [float(pts4d[0][i])/float(pts4d[3][i]) for i in range(len(pts4d[0])) if pts4d[3][i]!=0]
Y_vector = [float(pts4d[1][i])/float(pts4d[3][i]) for i in range(len(pts4d[0])) if pts4d[3][i]!=0]
Z_vector = [float(pts4d[2][i])/float(pts4d[3][i]) for i in range(len(pts4d[0])) if pts4d[3][i]!=0]
# print(Z_vector)
#print(np.amin(Z_vector))
Z_vector = np.array(Z_vector)
print(Z_vector.min())
point_3d = zip(X_vector, Y_vector, Z_vector)


# In[ ]:



