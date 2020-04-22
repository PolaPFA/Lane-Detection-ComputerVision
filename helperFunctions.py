import numpy as np

def get_gaussian_filter(size, sigma = 1):
    center_x = size[1] / 2
    center_y = size[0] / 2
    gauss = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            gauss[i][j] = (1 / (2 * np.pi * np.square(sigma))) * np, pow(np.exp(), -(
                        np.square(i - center_y) + np.square(j - center_x)) / 2 * np.square(sigma))

    return gauss

def apply_filter(img, kernel):
    offset = len(kernel) // 2
    newImage = np.zeros(img.shape())
    for i in range(offset, img.size[0] - offset):
        for j in range(offset, img.size[1] - offset):
            newImage[i, j] = np.dot(img[i-offset:i+offset, j-offset:j+offset], kernel)
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