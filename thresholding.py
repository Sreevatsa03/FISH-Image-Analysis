"""
Thresholding (Part 2 by Joseph and Soumili)
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Converts file to an image    
def file_to_image(filename):
    return cv2.imread(filename)

def image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Representing the output as a plot
def plot_image(image, title = ''):
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img, 'gray', vmin = 0, vmax = 255)
    plt.title(title)
    plt.axis('off')
    plt.savefig(('thresholding_output'.strip() + str(title).strip() + '.png'.strip()))

# watershed 
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

def watershed(filename, output = "plot"):
    img = file_to_image(filename)
    gray = image_to_grayscale(img)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(img, img, mask = thresh)
    
    if output == "plot":
        plot_image(masked, "Watershed")
    
    else:
        cv2.imshow("Watershed", masked)
        cv2.waitKey(0)
        cv2.imwrite("watershed.png", masked)

# gaussian blurring
#https://www.pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/

# kernel size is how blurry an image is (must be an odd number)
def gaussian_blur(filename, kernel_size):
    # read image
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
     
    # apply guassian blur on src image
    dst = cv2.GaussianBlur(src,(kernel_size, kernel_size),cv2.BORDER_DEFAULT)
     
    # display output image as pop up window
    cv2.imshow("Gaussian Smoothing", dst)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    cv2.imwrite("gaussian_blur.png", dst)
    

def binary_threshold(filename, threshold, output = "plot"):
    image = file_to_image(filename)
    gray = image_to_grayscale(image)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(image, image, mask=thresh)
    
    if output == "plot":
        plot_image(masked,'Binary Threshold')  
        
    else:
        cv2.imshow("Binary Threshold", masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("binary_threshold.png", masked)
        
    return thresh   
    
# Reference: https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/   
def erosion(filename, iterations, output = "plot"):
    img = cv2.imread(filename, 0)
    kernel = np.ones((5,5), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(img, kernel, iterations)
    img_dilation = cv2.dilate(img, kernel, iterations)
    
    if output == "plot":
        plot_image(img_erosion, "Erosion")
        plot_image(img_dilation, "Dilation")
     
    else:
        cv2.imshow('Input', img)
        cv2.imshow('Erosion', img_erosion)
        cv2.imshow('Dilation', img_dilation) 
        cv2.imwrite("erosion.png", img_erosion)
        cv2.waitKey(0)

# Reference: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
def get_centroids(image, threshold):
    centroids = []
    img = file_to_image(image)
    mask = np.zeros(img.shape, dtype=np.uint8)
    thresh = binary_threshold(image, threshold)
    
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:
        
        # calculate moments for each contour
        M = cv2.moments(c)
        
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
           
        else:
            cX, cY = int(c[0][0][0]), int(c[0][0][1])
           
        coord = (cX, cY)
        centroids.append(coord)
        
        cv2.circle(mask, (cX, cY), 1, (255, 255, 255), -1)
       
    #display the image
    cv2.imwrite('centroids.png', mask)
    cv2.waitKey(0)
    
    return centroids

if __name__ == "__main__":
    get_centroids('C2 (Pax6) thresholded dots.tif', 0)
    #get_centroids('MAX_C3-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif', 200)
    #watershed('MAX_C4-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif', "plot")
    #gaussian_blur('MAX_C4-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif', 11)
    #binary_threshold('MAX_C3-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif', 190)
    #erosion('MAX_C3-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif', 100)
    

