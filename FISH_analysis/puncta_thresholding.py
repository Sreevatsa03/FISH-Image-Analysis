import matplotlib.pyplot as plt
import numpy as np
import cv2


class Puncta_Thresholding:
    """
    Instantiate thresholing of puncta in spinal cord cell image\n
    Various types of threshoding can be performed on the given image

    :param img: FISH image
    :type img: .tif
    """

    def __init__(self, img):
        self.img = img

    def plot_image(self, image, title = ''):
        """
        Plot output of threshold as png

        :param image: output image to be plotted
        :param str title: name of saved png
        :type image: Any
        """

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(RGB_img, 'gray', vmin = 0, vmax = 255, aspect='auto')
        fig.savefig(('thresholding_output/'.strip() + str(title).strip().lower() + '.png'.strip()))

    def watershed(self, output = "plot") -> None:
        """
        Thresholding method - description needed

        :param str output: determines if output will just be plotted or will be displayed in a pop up window
        """

        # read and grayscale image
        img = cv2.imread(self.img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #  watershed image
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(img, img, mask = thresh)
        
        # output result of watersheding
        if output == "plot":
            self.plot_image(masked, "Watershed")
        else:
            # display image in pop up window
            cv2.imshow("Watershed", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 
            cv2.imwrite("thresholding_output/watershed.png", masked)

    def gaussian_blur(self, kernel_size, output = "plot"):
        """
        Thresholding method - description needed

        :param kernel_size: determines how blurry an image is
        :param str output: determines if output will just be plotted or will be displayed in a pop up window
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

    def binary_threshold(self, threshold, output = "plot"):
        """
        Thresholding method - description needed

        :param kernel_size: determines how blurry an image is
        :param str output: determines if output will just be plotted or will be displayed in a pop up window
        :type kernel_size: odd integer
        """
        
        image = cv2.imread(self.img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(image, image, mask=thresh)
        
        if output == "plot":
            self.plot_image(masked,'Binary_Threshold')
        elif output == "none":
            return thresh
        else:
            cv2.imshow("Binary Threshold", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("thresholding_output/binary_threshold.png", masked)
            
        return thresh   
  
    def erosion(self, iterations, output = "plot"):
        img = cv2.imread(self.img, 0)
        kernel = np.ones((5, 5), np.uint8)
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
            cv2.imwrite("thresholding_output/erosion.png", img_erosion)
            cv2.waitKey(0)

    def get_centroids(self, threshold):
        centroids = []
        img = cv2.imread(self.img)
        mask = np.zeros(img.shape, dtype=np.uint8)
        thresh = self.binary_threshold(threshold, 'none')
        
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
        cv2.imwrite('thresholding_output/centroids.png', mask)
        cv2.waitKey(0)
        
        return centroids