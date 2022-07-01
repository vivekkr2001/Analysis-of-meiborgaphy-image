from random import random as rand
from random import randint
import re
from turtle import width
from unittest import result
from scipy.interpolate import interp1d
import cv2
from cv2 import invert
import numpy as np

def binarise (img, threshold=100) :
    print(img.shape)
    binarise_image = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i][j] < threshold):
                   img[i][j] = 0
            else:
                   img[i][j] = 255       
    binarise_image = img
    return binarise_image

def dilation(image):
    kernel = np.ones((5,5), np.uint8)  
    dilated_image = cv2.dilate(image, kernel, iterations=4)  
    return dilated_image

def subtracted(original_image, processed_image):
    subtracted_image = cv2.subtract(original_image, processed_image)
    return subtracted_image

def median_filter(noisy_image): 
    m, n = noisy_image.shape
    filtered_image = np.zeros([m, n])
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [noisy_image[i-1, j-1],
                noisy_image[i-1, j],
                noisy_image[i-1, j + 1],
                noisy_image[i, j-1],
                noisy_image[i, j],
                noisy_image[i, j + 1],
                noisy_image[i + 1, j-1],
                noisy_image[i + 1, j],
                noisy_image[i + 1, j + 1]]
            temp = sorted(temp)
            filtered_image[i, j]= temp[4]
            
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image

def convolution_filter(image):
    if image is None:
        print('Could not read image')
    
    kernel1 = np.array([[0, 0, 0],
	                    [0, 1, 0],
	                    [0, 0, 0]])

    convolution_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
    return convolution_image        

def binary_inv(image):
    inverted = cv2.bitwise_not(image)
    return inverted

def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

def gradient_convex(src_gray, val):
    threshold = val
    retVal, mask = cv2.threshold(src_gray,155,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    canny_output = cv2.Canny(gradient, threshold, 2*threshold)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    drawing = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (0, 255, 0)
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
        # cnt = contours[i]
        # cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 3)
        cv2.fillConvexPoly(drawing, contours[i],(0, 255, 0), lineType=4, shift=0)
    # cv2.imshow('contour', drawing)

    return drawing


def laplace_filter(source_image):
    ddepth = cv2.CV_16S
    kernel_size = 7

    destination_image = cv2.Laplacian(source_image, -3, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(destination_image)
    
    return abs_dst

def binarise_invert(image):
    binarise_image2 = binarise(image)
    bin_inv_image = binary_inv(binarise_image2)
    return bin_inv_image
    
def border_reject(subtracted_image):
    kernel = np.ones((5,5), np.uint8)  
    img_erosion = cv2.erode(subtracted_image, kernel, iterations=1)  
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)  

    gX = cv2.Sobel(img_dilation, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gY = cv2.Sobel(img_dilation, ddepth=cv2.CV_32F, dx=0, dy=1, ksize= -1)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    return combined

# def largest_item(cont):
#     kernel = np.ones((5,5), np.uint8)  
#     erosion_img = cv2.erode(cont, kernel, iterations=1) 
#     img_dilation = cv2.dilate(erosion_img, kernel, iterations=1) 
#     cv2.imshow('dil Object', img_dilation) 
#     sorted_contours = get_contour_areas(contours = contours) 
#     sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
#     largest_item= sorted_contours[0]

#     cv2.drawContours(erosion_img, largest_item, -1, (255,0,0),10)

if __name__ == "__main__":
    image = cv2.imread('/home/vivek/security/c++_learning/Screenshot from 2022-06-24 14-45-42.png')
    
    img_gaussian = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    binarise_image = binarise(img_prewittx + img_prewitty)
    dilated_image = dilation(binarise_image)
    subtracted_image = subtracted(img_gaussian, dilated_image)
    filtered_image = median_filter(subtracted_image)
    convolution_image = convolution_filter(filtered_image)
    laplace_image = laplace_filter(convolution_image)
    bin_inv_image = binarise_invert(laplace_image)
    subtracted_image2 = subtracted(img_gaussian, bin_inv_image)
    # cv2.imshow('Sub_img', subtracted_image2)
    border_reject_image = border_reject(subtracted_image2)
    # cv2.imshow('combined', border_reject_image)
    result_image = gradient_convex(border_reject_image, 150)

    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()