import matplotlib.pyplot as plt
import matplotlib.lines as matline
import numpy as np
import cv2

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

    return hough_accum, thetas, rho

def get_hough_lines(accum, thetas, rho, img_size):
    maximum_number = np.max(accum)
    #r = 1500
    least_maximum = maximum_number - (maximum_number*0.2)
    width,height = img_size
    #thetas = np.rad2deg(thetas)
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
                x0 = (a * rho_val)
                y0 = (b * rho_val)
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                #pt2 = (int(x0 - width * (-b)), int(y0 - height * (a)))

                indices.append((pt1,pt2))

            acc[max_index_row, max_index_col] = 0
        else:
            break
    indices = np.array(indices, dtype=int)
    return indices

def show_hough_line(img, accumulator, thetas, rhos):

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

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


image = plt.imread('george.jpg')
image = cv2.imread('george.jpg')
print(image.shape)
hough_image = image[:,:,0]
x_indx, y_indx = np.where(hough_image == 255)
hough_accum, thetas, rho = hough_transform(hough_image)
show_hough_line(hough_image, hough_accum, thetas, rho)
lines = get_hough_lines(hough_accum, thetas, rho, hough_image.shape)

#figure = plt.figure(figsize=(10,10))

#subfig = figure.add_subplot(1,1,1)


#plt.imshow(image)
for line in lines:
    x1,y1 = line[0]
    x2,y2 = line[1]
    print(line[0])
    print(line[1])
 #   cv2.line(image, (y1, x1), (y2, x2), (0, 0, 255), 2)
#cv2.imshow("test",image)



print("end")

lines = cv2.HoughLines(hough_image,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    print((x1,y1))
    print((x2,y2))

    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow("test",image)
cv2.waitKey(0)

