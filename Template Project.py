import matplotlib.pyplot as plt
import helperFunctions
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate

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
def select_white(image):
    # white color mask
    lower = np.uint8([200  , 200,   200])
    upper = np.uint8([255, 250, 250])

    white_mask = inRange(image, lower, upper)


    return white_mask
def inRange(image, lower, upper):



    newImage = np.zeros(image.shape )
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp =image[i,j]
            if (temp[0]>=lower[0] and temp[0]<=upper[0] )and( temp[1]>=lower[1] and temp[1]<=upper[1]) and(temp[2]>=lower[2]  and temp[2]<=upper[2]):
                newImage[i,j]=[1,1,1]

    return newImage[:,:,0]

#main part

#1 read the image
image = plt.imread('test1.jpg')
plt.imshow(image)
plt.show()
print(image.shape)
#2 convert to HSV
hsv_image = convert_rgb_to_hsv(image.copy())


#3 convert to Gray
gray_image = convert_rbg_to_grayscale(image)
#plt.imshow(gray_image, cmap='gray')
#plt.show()
#4 Threshold HSV for Yellow and White (combine the two results together)

gray_image_thresh = gray_image > 230
gray_image_thresh[:700,:] = False
#plt.imshow(gray_image_thresh, cmap='gray')
#plt.show()

white_thrsh=select_white(image)
hsv_image_copy = hsv_image.copy()

hsv_image_thresh = hsv_image[:,:,2] > 200
hsv_image_copy[:,:,0] = hsv_image_copy[:,:,0]*hsv_image_thresh
hsv_image_copy[:,:,1] = hsv_image_copy[:,:,1]*hsv_image_thresh
hsv_image_copy[:,:,2] = hsv_image_copy[:,:,2]*hsv_image_thresh
hsv_image_thresh = hsv_image_copy[:,:,1] > 120
hsv_image_thresh[:,1000:] = False
#plt.imshow(hsv_image_thresh)
#plt.show()
plt.imshow(hsv_image_thresh)
plt.show()
plt.imshow(white_thrsh)
plt.show()
final_thresh = hsv_image_thresh + white_thrsh
plt.imshow(final_thresh)
plt.show()

#5 Mask the gray image using the threshold output fro step 4
#6 Apply noise remove (gaussian) to the masked gray image
#7 use canny detector and fine tune the thresholds (low and high values)
#8 mask the image using the canny detector output
#9 apply hough transform to find the lanes
#10 apply the pipeline you developed to the challenge videos

#11 You should submit your code
