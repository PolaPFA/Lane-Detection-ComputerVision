import numpy as np
import  math

def get_gaussian_filter(size, sigma = 1):
    center_x = size[1] / 2
    center_y = size[0] / 2
    gauss = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
           # gauss[i,j] = (1 / (2 * np.pi * np.square(sigma))) * pow(np.exp(), -( np.square(i - center_y) + np.square(j - center_x)) / 2 * np.square(sigma))
           temp1=1/(2*np.pi*sigma*sigma)
           temp=-( np.square( center_y-i) + np.square( center_x-j) / 2 * np.square(sigma))
           temp2=np.exp(temp)

           gauss[i][j]=temp1*temp2
    return gauss

def apply_filter(img, kernel):

    if len(kernel)%2==0:
        l=len(kernel)
        offset = len(kernel) // 2
        newImage = np.zeros(img.shape)
        for i in range(offset, img.shape[0] - offset):
            for j in range(offset, img.shape[1] - offset):
                temp = img[i - offset:i + offset, j - offset:j + offset]
                newImage[i, j] = sum(sum(np.dot(temp, kernel)))
    else:
        offset = int(len(kernel)/ 2)
        newImage = np.zeros(img.shape)
        for i in range(offset, img.shape[0] - offset):
            for j in range(offset, img.shape[1] - offset):
                temp = img[i - offset:i + offset+1, j - offset:j +offset+1]
                newImage[i, j] = sum(sum(np.dot(temp, kernel)))



    return newImage

def sobel_filter(img, direction):
    if direction == 0:
        filter = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        return apply_filter(img, filter)
    else:
        filter = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        return apply_filter(img, filter)

def non_maxima(img, theta):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            before = 255
            after = 255
            try:
                if theta[i, j] >= 0 & theta[i, j] < 22.5:
                    before = img[i, j+1]
                    after = img[i, j-1]
                elif theta[i, j] >= 22.5 & theta[i, j] < 67.5:
                    before = img[i-1, j + 1]
                    after = img[i+1, j - 1]
                elif theta[i, j] >= 67.5 & theta[i, j] < 112.5:
                    before = img[i - 1, j ]
                    after = img[i + 1, j]
                elif theta[i, j] >= 112.5 & theta[i, j] < 157.5:
                    before = img[i - 1, j - 1]
                    after = img[i + 1, j + 1]
            except IndexError:
                continue
            if before > img[i, j] | after > img[i, j]:
                img[i, j] = 0
    return img

def double_threshold(img, low, high):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            if img[i, j] < low:
                img[i, j] = 0
            elif img[i, j] < high:
                img[i, j] = 25
            else:
                img[i, j] = 255
    return img

def hysteresis_edge_tracking(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            start_i = np.maximum(0, i-1)
            end_i = np.minimum(255, i+1)
            start_j =  np.maximum(0, j-1)
            end_j = np.minimum(255, j+1)
            mask = np.ones([3,3])
            outp = np.dot(mask,img[start_i:end_i, start_j:end_j])
            if outp > 255:
                img[i, j] = 255
            else:
                img[i,j] = 0
    return img
def select_white(image):
    # white color mask
    lower = np.uint8([200  , 0,   190])
    upper = np.uint8([255, 255, 255])

    white_mask = inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([230,100,30])
    upper = np.uint8([255, 210,110])
    yellow_mask = inRange(image, lower, upper)


    return (white_mask,yellow_mask)
def inRange(image, lower, upper):



    newImage = np.zeros(image.shape )
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp =image[i,j]
            if (temp[0]>=lower[0] and temp[0]<=upper[0] )and( temp[1]>=lower[1] and temp[1]<=upper[1]) and(temp[2]>=lower[2]  and temp[2]<=upper[2]):
                newImage[i,j]=[1,1,1]

    return newImage[:,:,0]


def hough_transform(image, angles=np.linspace(-90,90, 181)):
    thetas = np.deg2rad(angles)
    img_width = image.shape[0]
    img_height = image.shape[1]
    diagonal = int(np.sqrt(np.square(img_width)+np.square(img_height)))
    rho = np.linspace(-diagonal, diagonal, diagonal*2)

    cosines = np.cos(thetas)
    sines = np.sin(thetas)
    hough_accum = np.zeros((2*diagonal, len(thetas)))

    x_index, y_index = np.nonzero(image)
    for i in range(len(x_index)):
        x = x_index[i]
        y = y_index[i]

        for theta in range(len(thetas)):
            rho_val = int(diagonal + int(x * cosines[theta] + y * sines[theta]))
            hough_accum[rho_val, theta] += 1
    return hough_accum, thetas , rho

def get_hough_lines(accum, thetas, rho):
    maximum_number = np.max(accum)
    #r = 1500
    least_maximum = maximum_number - (maximum_number*0.5)

    acc = accum.copy()
    indices = []
    while (True):
        temp_max = np.max(acc)
        if (temp_max >= least_maximum):
            max_index_row, max_index_col = np.where(acc == temp_max)
            for idx_r, idx_c in zip(max_index_row, max_index_col):
                theta_val = thetas[idx_c]
                rho_val = rho[idx_r]

                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho_val
                y0 = b * rho_val
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))


                indices.append((pt1,pt2))

            acc[max_index_row, max_index_col] = 0
        else:
            break
    indices = np.array(indices, dtype=int)
    return indices

def show_hough_line(img, accumulator, thetas, rhos):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    plt.show()

