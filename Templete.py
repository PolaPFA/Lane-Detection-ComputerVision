import matplotlib.pyplot as plt
import helperFunctions
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate
from skimage import feature

#import cv2

team_members_names = ['بلال هاني كمال', 'بولا فرج أسعد', 'بيتر ماجد منير', 'جورج كميل برسوم', 'جون اميل يوحنا']
team_members_seatnumbers = ['2016170130', '2016170133', '2016170134', '2016170144', '2016170146']


def draw_lines_connected(img, lines, color=[255, 0, 0], thickness=8):
    pass
    # this function should draw lines to the images (default color is red and thickness is 8)


def convert_rbg_to_grayscale(img):
    out_image = np.zeros([img.shape[0], img.shape[1]])
    out_image[:, :] = np.sum(img[:, :, :], axis=2) / img.shape[2]
    out_image = out_image.astype('float32')
    return out_image


def convert_rgb_to_hsv(img):
    r_dash = img[:, :, 0] / 255
    b_dash = img[:, :, 1] / 255
    g_dash = img[:, :, 2] / 255

    out_image = np.zeros([img.shape[0], img.shape[1], 3])
    c_max = np.max([r_dash, b_dash, g_dash], axis=0)
    c_min = np.min([r_dash, b_dash, g_dash], axis=0)
    delta = c_max - c_min
    out_image[:, :, 2] = c_max * 255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if c_max[i, j] != 0:
                out_image[i, j, 1] = delta[i, j] / c_max[i, j] * 255
            if c_max[i, j] == r_dash[i, j]:
                out_image[i, j, 0] = 60 * np.mod((g_dash[i, j] - b_dash[i, j]) / delta[i, j], 6)
            elif c_max[i, j] == g_dash[i, j]:
                out_image[i, j, 0] = 60 * (((b_dash[i, j] - r_dash[i, j]) / delta[i, j]) + 2)
            elif c_max[i, j] == b_dash[i, j]:
                out_image[i, j, 0] = 60 * (((r_dash[i, j] - g_dash[i, j]) / delta[i, j]) + 4)
    out_image = out_image.astype('int64')
    return out_image
    # This function will do color transform from RGB to HSV


def detect_edges_canny(img, low_threshold, high_threshold):
    denoised_image = remove_noise(img, 3)
    vertical_image = helperFunctions.sobel_filter(denoised_image, 0)
    horizontal_image = helperFunctions.sobel_filter(denoised_image, 1)
    sobeled_image = np.hypot(vertical_image, horizontal_image)
    thetas = np.arctan2(horizontal_image, vertical_image)
    non_maxima_img = helperFunctions.non_maxima(sobeled_image, thetas)
    return helperFunctions.hysteresis_edge_tracking(non_maxima_img)
    # You should implement yoru Canny Edge Detector here


def remove_noise(img, kernel_size):
    gauss_kernel = helperFunctions.get_gaussian_filter(kernel_size)
    new_image = helperFunctions.apply_filter(img, gauss_kernel)
    return new_image
    # You should implement Gaussian Noise Removal Here


def mask_image(img, vertices):
    for i in range(vertices.shape[0]):
        outside = True
        for j in range(vertices.shape[1]):
            if outside & vertices[i, j] == 0:
                img[i, j] == 0
            elif vertices[i, j] == 255:
                outside = ~outside
    return img
    # Mask out the pixels outside the region defined in vertices (set the color to black)


# main part

# 1 read the image
image = plt.imread('test.jpg')
print('Original Image')
plt.imshow(image)
print(image.shape)
#plt.show()
# 2 convert to HSV
hsv_image = convert_rgb_to_hsv(image)
print('HSV Image')
plt.imshow(hsv_image)
#plt.show()
# 3 convert to Gray
gray_image = convert_rbg_to_grayscale(image)
#plt.imshow(gray_image)
#plt.show()
# 4 Threshold HSV for Yellow and White (combine the two results together)
(white,yellow)= helperFunctions.select_white(image)
print('White Image')
plt.imshow(white)
#plt.show()
print('Yellow Image')
#plt.imshow(yellow)
#plt.show()
# 5 Mask the gray image using the threshold output fro step 4
newgray=(gray_image*white)+(gray_image*yellow)

print('Masked Image')
#plt.imshow(newgray)
#plt.show()
# 6 Apply noise remove (gaussian) to the masked gray image
print('Gussian')
size=newgray.shape
gauss=helperFunctions.get_gaussian_filter((10,10),1)
newgray=helperFunctions.apply_filter(newgray,gauss)
plt.imshow(newgray)
plt.show()
# 7 use canny detector and fine tune the thresholds (low and high values)
print('Edge detection')
#ed1=helperFunctions.sobel_filter(newgray, 0)
#ed2=helperFunctions.sobel_filter(newgray, 1)
#ed=ed1+ed2
ed=helperFunctions.canny(newgray)
plt.imshow(ed)
plt.show()

# 8 mask the image using the canny detector output
print('Mask')
for i in range( ed.shape[0] ):
    for j in range(ed.shape[1]):
        if ed[i][j]!=0:
            ed[i,j]=1
newgray=newgray*ed
# 9 apply hough transform to find the lanes

hough_image=newgray
print('Hough Image')
plt.imshow(hough_image)
plt.show()

hough_accum, thetas, rho = helperFunctions.hough_transform(hough_image)
lines = helperFunctions.get_hough_lines(hough_accum, thetas, rho)
#print(lines)
#print('Done')
count=0
plt.imshow(image)
for line in lines:
    l1 = line[0]
    x1=l1[0]
    y1=l1[1]

    l2 = line[1]
    x2=l2[0]
    y2=l2[1]
    plt.plot((x1,x2), (y1,y2))
    #cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0),5)
#cv2.imwrite(str(count)+'.jpg', image)
plt.show()
# 10 apply the pipeline you developed to the challenge videos

# 11 You should submit your code