import matplotlib.pyplot as plt
import numpy as np
import cv2

class Puncta_Thresholding:

    def __init__(self, filename):
        self.filename = filename

    # Converts file to an image    
    def file_to_image(self):
        return cv2.imread(self.filename)

    def image_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Representing the output as a plot
    def plot_image(self, image, title = ''):
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img, 'gray', vmin = 0, vmax = 255)
        plt.title(title)
        plt.axis('off')
        plt.savefig(('thresholding_output'.strip() + str(title).strip() + '.png'.strip()))

    # def save_image(image):
    #     # TODO: figure out how to save imagex
    #     # Image path
    #     # image = cv2.imread('C:/Users\19787\.spyder-py3\DS 2500\thresholding\image1.png')
        
    #     # # Image directory
    #     # directory = r'C:\Users\19787\Desktop'

    #     # img = cv2.imread('D:/image-1.png')
    #     #do some transformations on img
        
    #     #save matrix/array as image file
    #     isWritten = cv2.imwrite('D:/image-2.png', image)
        
    #     if isWritten:
    #         print('Image is successfully saved as file.')
            
    # watershed 
    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

    def watershed(self, output = "plot"):
        img = self.file_to_image(self.filename)
        gray = self.image_to_grayscale(img)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(img, img, mask = thresh)
        
        if output == "plot":
            self.plot_image(masked, "Watershed")
        
        else:
            cv2.imshow("Watershed", masked)
            cv2.waitKey(0)

    # gaussian blurring
    #https://www.pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/

    # kernel size is how blurry an image is (must be an odd number)
    def gaussian_blur(self, kernel_size):
        # read image
        src = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        
        # apply guassian blur on src image
        dst = cv2.GaussianBlur(src,(kernel_size, kernel_size),cv2.BORDER_DEFAULT)
        
        # display output image as pop up window
        image = cv2.imshow("Gaussian Smoothing", dst)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image
        #save_image(image)

    def binary_threshold(self, threshold, output = "plot"):
        image = self.file_to_image(self.filename)
        gray = self.image_to_grayscale(image)
        ret, thresh1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(image, image, mask=thresh1)
        
        if output == "plot":
            self.plot_image(masked,'Binary Threshold')  
            #save_image(masked, name = 'TEST')
            
        else:
            cv2.imshow("Binary Threshold", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # Reference: https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/   
    def erosion_dilation(self, iterations, output = "plot"):
        img = cv2.imread(self.filename, 0)
        kernel = np.ones((5,5), np.uint8)
        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        img_erosion = cv2.erode(img, kernel, iterations)
        img_dilation = cv2.dilate(img, kernel, iterations)
        
        if output == "plot":
            self.plot_image(img_erosion, "Erosion")
            self.plot_image(img_dilation, "Dilation")
        
        else:
            cv2.imshow('Input', img)
            cv2.imshow('Erosion', img_erosion)
            cv2.imshow('Dilation', img_dilation) 
            cv2.waitKey(0)