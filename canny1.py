from random import random as rand
from random import randint
import re
from turtle import width
from unittest import result
from scipy.interpolate import interp1d
import cv2
from cv2 import invert
import numpy as np

def binarise (img, threshold=80) :
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
    dilated_image = cv2.dilate(image, kernel, iterations=2)  
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

def remove_small_obj(image, size):
    blur = cv2.GaussianBlur(image, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < size:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    return close

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
    
    contours, _ = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    drawing = np.ones((gradient.shape[0], gradient.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (0, 255, 0)
        cv2.drawContours(drawing, hull_list, i, color)
        cv2.drawContours(drawing, contours, i, color)
        cnt = contours[i]
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 1)
        cv2.fillConvexPoly(drawing, contours[i],(0, 255, 0), lineType=4, shift=0)

    kernel = np.ones((5,5), np.uint8)
    erosion_img = cv2.erode(drawing, kernel, iterations=1) 
    sorted_contours = get_contour_areas(contours = contours) 
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    largest_item= sorted_contours[0]
    cv2.drawContours(erosion_img, largest_item, -1, (255,0,0),1)

    img_dilation = cv2.dilate(erosion_img, kernel, iterations=1) 
    gray_img = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)
    binarise_result = binarise(gray_img)
    
    return binarise_result


def laplace_filter(source_image, kernel_size):
    ddepth = cv2.CV_16S

    destination_image = cv2.Laplacian(source_image, -3, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(destination_image)
    
    return abs_dst

def binarise_invert(image):
    #Binarise
    binarise_image2 = binarise(image)
    #Inverse-binarisation
    bin_inv_image = binary_inv(binarise_image2)
    return bin_inv_image
    
def border_reject(subtracted_image):
    kernel = np.ones((5,5), np.uint8)
    #Erosion  
    img_erosion = cv2.erode(subtracted_image, kernel, iterations=1) 
    #Dilation
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    #Removing small objects
    # small = remove_small_obj(img_dilation, 500)  
    #Border Rejecting
    gX = cv2.Canny(img_dilation, 40, 200)
    gY = cv2.Canny(img_dilation, 40, 200)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    return combined

def remove_angle(image):
    if image is None:
        print('Could not read image')
    #45 degree
    kernel1 = np.array([[2, -1, -1],
	                    [-1, 2, -1],
	                    [-1, -1, 2]])

    angle_image1 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
    #135 degree
    kernel2 = np.array([[-1, -1, 2],
	                    [-1, 2, -1],
	                    [2, -1, -1]])

    angle_image2 = cv2.filter2D(src=angle_image1, ddepth=-1, kernel=kernel2)
    return angle_image2  

if __name__ == "__main__":
    image = cv2.imread('/home/vivek/security/image-processing/Processed data/2nagarth/left.png')
    
    #Algorithms to extract Region of interest 
    cv2.imshow("Original",image)
    #Prewitt, Binarizing and Dilating #I-M
    img_gaussian = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    binarise_image = binarise(img_prewittx + img_prewitty, 40)
    dilated_image = dilation(binarise_image)
    # cv2.imshow("step-1",dilated_image)
    #Subtracting I-M from original image
    subtracted_image = subtracted(img_gaussian, dilated_image)
    # cv2.imshow("step-2",subtracted_image)

    #Median-filter, convolution highlight detail filter and laplacian filter
    filtered_image = median_filter(subtracted_image)
    convolution_image = convolution_filter(filtered_image)
    laplace_image = laplace_filter(convolution_image, 15)
    # cv2.imshow("step-3",laplace_image)

    #Binarise and binary inversing #I-BI
    bin_inv_image = binarise_invert(laplace_image)
    # cv2.imshow("step-4",bin_inv_image)

    #Subtracting I-M from I-BI
    sub_image = subtracted(img_gaussian, bin_inv_image)
    # cv2.imshow("step-5",sub_image)

    #Eroding,dilating, removing small objects and border rejecting
    border_reject_image = border_reject(sub_image)
    # cv2.imshow("step-6",border_reject_image)

    #Gradient out and convex hull, eroding keeping largest object, dilating and converting to gray and curve fitting
    result_image = gradient_convex(border_reject_image, 100)
    result_image = remove_small_obj(result_image, 5000)
    #Calculation of pixel constituting Region of interest
    cv2.imshow('Result', result_image)
    cv2.imwrite("/home/vivek/security/image-processing/Processed data/2nagarth/left-roi.png",result_image)
    number_of_white_pix = np.sum(result_image == 255)
    print("ROI Pixel : ", number_of_white_pix)
    
#########################################################################
    #Algos to segment gland #I-GHD
    img_2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fil_img = median_filter(img_2_gray)
    convoluted_img = convolution_filter(fil_img)
    lap_img = laplace_filter(convoluted_img, 15)
    # cv2.imshow("lappp",lap_img)
    # multiplying with ROI and binarizing
    cropped_img = cv2.multiply(lap_img, result_image)
    bin_img = binarise(cropped_img,60)
    # cv2.imshow("lappp",cropped_img)

    # median filtering and removing objects with angle 45 and 135 in degrees.
    fil_crop_img = median_filter(bin_img)
    kernel = np.ones((5,5), np.uint8) 
    fil_crop_img = cv2.erode(fil_crop_img, kernel,iterations=1) 
    fil_crop_img = cv2.dilate(fil_crop_img, kernel, iterations=3)    
    angle_img = remove_angle(fil_crop_img)
    subtract_img = subtracted(fil_crop_img, angle_img)
    subtract_img = subtracted(result_image, subtract_img)
    # subtract_img = remove_small_obj(subtract_img, 5500)
    #Calculation of pixel constituting gland size
    cv2.imshow("Gland Image",subtract_img)
    cv2.imwrite("/home/vivek/security/image-processing/Processed data/2nagarth/left-gland.png",subtract_img)
    number_of_gland_pix = np.sum(subtract_img == 255)
    print("Gland Pixel: ",number_of_gland_pix)

    percentage = (number_of_gland_pix * 100) / number_of_white_pix
    print("Percentage: ",percentage)
    grade = 3
    if percentage > 47:
        grade = 0
    elif percentage > 40 :
        grade = 1
    elif percentage > 31:
        grade = 2
    print("Grade: ",grade)
    file = open('/home/vivek/security/image-processing/Processed data/2nagarth/left-data.txt', 'w')
    file.write("Total region of interest pixel: {} \nGland pixel: {}\nPercentage: {}\nGrade: {}".format(number_of_white_pix,number_of_gland_pix,percentage,grade))
    file.close()
    key = cv2.waitKey(0)
    # if(key == 27):
    #     cv2.destroyAllWindows()