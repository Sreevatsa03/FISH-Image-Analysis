import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Any


class Puncta_Thresholding:
    """
    Instantiate thresholding of puncta in spinal cord cell image\ 
    Various types of thresholding can be performed on the given image

    :param img: FISH image
    :type img: .tif
    """

    def __init__(self, img) -> None:
        self.img = img

    def plot_image(self, image, title='') -> None:
        """
        Plot output of threshold as png

        :param image: output image to be plotted
        :param str title: name of saved png
        :type image: array of image
        """

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(RGB_img, 'gray', vmin = 0, vmax = 255, aspect='auto')
        fig.savefig(('thresholding_output/'.strip() + str(title).strip().lower() + '.png'.strip()))

    def watershed(self, output='plot') -> None:
        """
        Produces an image that is thresholded based on region-based image morphology to determine the foreground and background

        :param str output: determines if output will just be plotted,
            or will be displayed in a pop up window
        """

        # read and grayscale image
        img = cv2.imread(self.img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #  watershed image
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(img, img, mask = thresh)
        
        # output result of watersheding
        if output == "plot":
            self.plot_image(masked, "Watershed")
        else:
            # display image in pop up window
            cv2.imshow("Watershed", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/watershed.png", masked)

    def gaussian_blur(self, kernel_size, output='plot') -> None:
        """
        Produces a blurred image that is based on the inputted kernel size to reduce image noise and reduce detail

        :param kernel_size: determines how blurry an image is
        :param str output: determines if output will just be plotted,
            or will be displayed in a pop up window
        :type kernel_size: odd integer
        """

        # read image
        src = cv2.imread(self.img, cv2.IMREAD_UNCHANGED)
        
        # apply guassian blur on src image
        dst = cv2.GaussianBlur(src, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

        # output result of blurring
        if output == "plot":
            cv2.imwrite("thresholding_output/gaussian_blur.png", dst)
        else:
            # display output image in pop up window
            cv2.imshow("Gaussian Smoothing", dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/gaussian_blur.png", dst)

    def binary_threshold(self, threshold=190, output='plot') -> Any:
        """
        Produces a binary image based on an inputted threshold to determinine the background and foreground

        :param int threshold: threshold value used to classify pixel values,
            and filter out pixel values greater than given threshold value
        :param str output: determines if output will just be plotted,
            or just return image array,\ 
            or will be displayed in a pop up window
        :return thresh: thresholded image
        :rtype: array
        """
        
        # read and grayscale image
        image = cv2.imread(self.img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold and reaplace pixels in image
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(image, image, mask=thresh)
        
        # output result of binary thresholding
        if output == "plot":
            self.plot_image(masked,'Binary_Threshold')
        elif output == "none":
            # return thresholded image
            return thresh
        else:
            # display output image in pop up window
            cv2.imshow("Binary Threshold", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/binary_threshold.png", masked)
        
        # return thresholded image
        return thresh   
    
    def erosion_dilation(self, iterations=100, output='plot') -> None:
        """
        Erosion - Produces an image by removing small-scale details from a binary image to remove a layer of pixels from both the inner and outer boundaries of regions\ 
        Dilation - Produces an image by adding small-scale details to a binary image to add a layer of pixels to both the inner and outer boundaries of regions

        :param int iterations: number of iterations,
            which will determine how much to erode/dilate image
        :param str output: determines if output will just be plotted,
            or will be displayed in a pop up window
        """

        # read image into array used by cv2
        img = cv2.imread(self.img, 0)

        # matrix with which image is convolved
        kernel = np.ones((5, 5), np.uint8)

        # erode and dilate image
        img_erosion = cv2.erode(img, kernel, iterations)
        img_dilation = cv2.dilate(img, kernel, iterations)
        
        # output result of erosion and dilation
        if output == "plot":
            self.plot_image(img_erosion, "Erosion")
            self.plot_image(img_dilation, "Dilation")
        else:
            # display output image in pop up window
            cv2.imshow('Erosion', img_erosion)
            cv2.imshow('Dilation', img_dilation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/erosion.png", img_erosion)
            cv2.imwrite("thresholding_output/dilation.png", img_dilation)

    def get_centroids(self, threshold=177, outline=False, output='none') -> list[tuple[int, int]]:
        """
        Produces a list of the coordinates of centroids in the given image, and used for both cell and dot centroids

        :param int threshold: threshold value passed into binary threshold, 
            and used to classify pixel values,\ 
            and filter out pixel values greater than given threshold value
        :param bool outline: determine if function being used for outline of cell mask
        :param str output: determines if output will just be plotted,
            or just return image array,\ 
            or will be displayed in a pop up window
        :return centroids: list of centroid coordinates
        :rtype: list[tuple[int, int]]
        """

        # create empty list of centroids
        centroids = []

        # read image into array used by cv2
        img = cv2.imread(self.img)

        # numpy arrat filled with 0s
        mask = np.zeros(img.shape, dtype=np.uint8)

        # image is binary thresholded if function not being used for outline else it is grayscaled
        if not outline:
            thresh = self.binary_threshold(threshold, 'none')
        else:
            thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find contours in binary image
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # iterate through contours
        for c in contours:
        
            # calculate moments for each contour
            M = cv2.moments(c)
        
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = int(c[0][0][0]), int(c[0][0][1])
            
            # append coordinate to list of centorid coordinates
            coord = (cX, cY)
            centroids.append(coord)
            
            # draw circle where centroid would be
            cv2.circle(mask, (cX, cY), 1, (255, 255, 255), -1)
        
        # output result of getting centroids
        if output == "plot":
            self.plot_image(mask, 'Centroids')
        elif output == "none":
            # return list of centroids
            return centroids
        else:
            # display output image in pop up window
            cv2.imshow("Centroids", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/centroids.png", mask)
        
        # return list of centroids
        return centroids