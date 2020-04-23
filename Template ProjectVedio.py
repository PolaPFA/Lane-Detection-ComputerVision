import matplotlib.pyplot as plt
import cv2
import helperFunctions
import os
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate
from skimage.color import rgb2hsv


team_members_names = ['بلال هاني كمال', 'بولا فرج أسعد', 'بيتر ماجد منير', 'جورج كميل برسوم', 'جون اميل يوحنا']
team_members_seatnumbers = ['2016170130', '2016170133', '2016170134', '2016170144', '2016170146']


def draw_lines_connected(img, lines, color=[255, 0, 0], thickness=8):
    pass
    #this function should draw lines to the images (default color is red and thickness is 8)

def convert_rbg_to_grayscale(img):
    out_image = np.zeros([img.shape[0], img.shape[1]])
    out_image[:, :] = np.sum(img[:,:,:], axis = 2)/img.shape[2]
    out_image= out_image.astype('float32')
    return out_image
    
def convert_rgb_to_hsv(img):
    r_dash = img[:,:,0]/255
    g_dash = img[:,:,1]/255
    b_dash = img[:,:,2]/255
    out_image = np.zeros([img.shape[0], img.shape[1], 3])
    c_max = np.max([r_dash, g_dash, b_dash], axis=0)
    c_min = np.min([r_dash, g_dash, b_dash], axis=0)
    delta = c_max-c_min
    out_image[:,:,2] = c_max*255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if c_max[i, j] !=0:
                out_image[i,j,1] = delta[i,j]/c_max[i,j] * 255
            if c_max[i,j] == r_dash[i,j]:
                out_image[i,j,0] = 60 * np.mod((g_dash[i,j]-b_dash[i,j])/delta[i,j],6)
            elif c_max[i,j] == g_dash[i,j]:
                out_image[i, j, 0] = 60 * (((b_dash[i, j] - r_dash[i, j]) / delta[i, j]) + 2)
            elif c_max[i,j] == b_dash[i,j]:
                out_image[i, j, 0] = 60 * (((r_dash[i, j] - g_dash[i, j]) / delta[i, j]) + 4)
    out_image = out_image.astype('int64')
    return out_image
    #This function will do color transform from RGB to HSV
    
def detect_edges_canny(img, low_threshold, high_threshold):
    denoised_image = remove_noise(img, 3)
    vertical_image = helperFunctions.sobel_filter(denoised_image, 0)
    horizontal_image = helperFunctions.sobel_filter(denoised_image, 1)
    sobeled_image = np.hypot(vertical_image, horizontal_image)
    thetas = np.arctan2(horizontal_image, vertical_image)
    non_maxima_img = helperFunctions.non_maxima(sobeled_image, thetas)
    return helperFunctions.hysteresis_edge_tracking(non_maxima_img)
    #You should implement yoru Canny Edge Detector here

def remove_noise(img, kernel_size):
    gauss_kernel = helperFunctions.get_gaussian_filter(kernel_size)
    new_image = helperFunctions.apply_filter(img,gauss_kernel)
    return new_image
    #You should implement Gaussian Noise Removal Here
    
def mask_image(img, vertices):
    for i in range(vertices.shape[0]):
        outside = True
        for j in range(vertices.shape[1]):
            if outside & vertices[i, j] == 0:
                img[i, j] == 0
            elif vertices[i, j] == 255:
                outside = ~outside
    return img
    #Mask out the pixels outside the region defined in vertices (set the color to black)


#main part

#1 read the image
#image = plt.imread('test.jpg')
cap= cv2.VideoCapture('new.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
outimg=[]
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    #plt.imshow(frame)
    #plt.show()
    #2 convert to HSV
    hsv_image = convert_rgb_to_hsv(frame)
    #plt.imshow(hsv_image)
    #plt.show()
    #3 convert to Gray
    gray_image = convert_rbg_to_grayscale(frame)
   # plt.imshow(gray_image)
   # plt.show()




    #4 Threshold HSV for Yellow and White (combine the two results together)
    #5 Mask the gray image using the threshold output fro step 4
    #6 Apply noise remove (gaussian) to the masked gray image
    #7 use canny detector and fine tune the thresholds (low and high values)
    #8 mask the image using the canny detector output
    #9 apply hough transform to find the lanes
    #10 apply the pipeline you developed to the challenge videos
    height, width, layers = frame.shape
    size = (width, height)
    outimg.append(frame)
    print(i)
    i+=1
    #11 You should submit your code

#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')


video_filename = 'output1.avi'
#out = cv2.VideoWriter(video_filename, fourcc, int(fps), width, height)
out = cv2.VideoWriter(video_filename,fourcc,fps,(width,height),True)

for img in outimg:
        #add this array to the video
    out.write(img)
    cv2.waitKey(1)
cap.release()
out.release()
cv2.destroyAllWindows()

print("end")