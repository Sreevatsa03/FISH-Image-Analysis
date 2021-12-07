import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image


class Puncta_Analysis:
    """

    Instantiate analysis of puncta in spinal cord cell image\n
    Various analyses can be performed on the given image
    
    :param mask: mask of FISH image
    :param dots: cleaned (thresholded dots) FISH image
    :type mask: .tif
    :type dots: .tif
    """
    
    def __init__(self, mask, dots) -> None:
        self.mask = mask
        self.dots = dots
        self.mask_edges
    
    def tif_to_png(self) -> None:
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
        ax.imshow(im_dots, aspect='auto', cmap='gray')
        fig.savefig('analysis_output/dots_plot.png')
        self.dots = 'analysis_output/dots_plot.png'
    
    def mask_edges(self) -> None:
        """
        Perform edge detection on mask\n
        Used in preparation for overlay of mask onto dots image
        """

        # read mask using opencv
        cv2_mask = cv2.imread(self.mask, 0)
        
        # detect edges and reverse black and white
        edges_detected = cv2.Canny(cv2_mask, 1, 1)
        ret, th2 = cv2.threshold(edges_detected, 100, 255, cv2.THRESH_BINARY_INV)
        
        # save plot of edges only mask
        plt.imshow(th2, cmap='spring')
        plt.axis('off')
        plt.savefig('analysis_output/mask_edges.png')
        self.mask_edges = 'analysis_output/mask_edges.png'
        
    def make_transparent(self) -> None:
        """
        Make edge_detected mask into image with transparent background\n
        Used in preparation for overlay of mask onto dots image
        """
        
        img = Image.open(self.mask_edges)
        img = img.convert("RGBA")
    
        data = img.getdata()
    
        newData = []
    
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
    
        img.putdata(newData)
        img.save('analysis_output/mask_edges.png', "PNG")
        self.mask_edges = 'Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001_otlines.png'
        
    def overlay(self) -> None:
        """
        Overlay mask with only edges on top of image of thresholded dots\n
        Will be able to perform analyses on outputed image\n
        Save overlay as image
        """

        # convert mask and dots to png
        self.tif_to_png()
        
        # get the edges_only mask and make transparent
        self.mask_edges()
        self.make_transparent()
        
        # open the image of the dots
        img1 = Image.open(self.dots)
        
        # open the image of the mask
        image = Image.open(self.mask_edges)
        resized_image = image.resize((500, 500))
        resized_image.save('analysis_output/outline.png')
        self.mask_edges = 'analysis_output/outline.png'

        image = Image.open(self.mask_edges)
        rotated = image.rotate(270, expand=True)
        rotated.save('analysis_output/outline.png')
        self.mask_edges = 'analysis_output/outline.png'

        img2 = Image.open(self.mask_edges)
        
        # paste the mask on top of the dots
        img1.paste(img2, (0,0), mask = img2)
        
        # save the image
        img1.save('analysis_output/overlay.png', "PNG")