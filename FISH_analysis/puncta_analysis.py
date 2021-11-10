import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


class Puncta_Analysis:
    """
    Instantiate analysis of spinal cord cell image\n
    Various analyses can be performed on the given image
    
    :param mask: mask of FISH image
    :param dots: cleaned (thresholded dots) FISH image
    :type mask: .tif
    :type dots: .tif
    """
    
    def __init__(self, mask, dots):
        self.mask = mask
        self.dots = dots
    
    def tif_to_png(self):
        """
        Convert given mask and dots files from .tif to .png\n
        Store converted mask as 'mask_plot.png' and dots as 'dots_plot.png'
        """
        
        # save plot of mask and store as self.mask
        im_mask = mpimg.imread(self.mask)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im_mask, aspect='auto')
        fig.savefig('analysis_output/mask_plot.png')
        self.mask = 'analysis_output/mask_plot.png'
        
        # save plot of dots and store as self.dots
        im_dots = mpimg.imread(self.dots)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im_dots, aspect='auto')
        fig.savefig('analysis_output/dots_plot.png')
        self.mask = 'analysis_output/dots_plot.png'
    
    def mask_edges(self, mask):
        """
        Perform edge detection on mask\n
        Used in preparation for overlay of mask onto dots image
        
        :param mask: mask of FISH image
        :type mask: .png
        """

        # read mask using opencv
        cv2_mask = cv2.imread(mask, 0)
        
        # detect edges and reverse black and white
        edges_detected = cv2.Canny(cv2_mask, 1, 1)
        ret, th2 = cv2.threshold(edges_detected, 100, 255, cv2.THRESH_BINARY_INV)
        
        # save plot of edges only mask
        plt.imshow(th2, cmap = 'gray')
        plt.axis('off')
        plt.savefig('analysis_output/mask_edges.png')