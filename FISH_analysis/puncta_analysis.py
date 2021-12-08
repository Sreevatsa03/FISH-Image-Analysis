import cv2
from PIL import Image
# from FISH_analysis import Puncta_Thresholding
import numpy as np
import PIL
from skimage.morphology import flood_fill
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import euclidean_distances
import os


class Puncta_Analysis:
    """

    Instantiate analysis of puncta in spinal cord cell image\n
    Various analyses can be performed on the given image
    
    :param outline: outline of cells from mask of FISH image,
        which can be created using Segmentation.make_outlines()
    :param dots: cleaned (thresholded dots) FISH image,
        which can be created using Puncta_Thresholding and its various thresholding functions (start with binary_threshold)
    :type outline: .png
    :type dots: .tif
    """
    
    def __init__(self, outline, dots) -> None:
        self.outline = outline
        self.dots = dots
        self.mask_edges
    
    def tif_to_png(self) -> None:
        """
        Convert given mask and dots files from .tif to .png\n
        Store converted mask as 'outline_plot.png' and dots as 'dots_plot.png'
        """
        
        # save plot of mask and store as self.mask
        im_mask = mpimg.imread(self.mask)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im_mask, aspect='auto')
        fig.savefig('analysis_output/outline_plot.png')
        self.outline = 'analysis_output/outline_plot.png'
        
        # save plot of dots and store as self.dots
        im_dots = mpimg.imread(self.dots)
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im_dots, aspect='auto', cmap='gray')
        fig.savefig('analysis_output/dots_plot.png')
        self.dots = 'analysis_output/dots_plot.png'
        
    def make_transparent(self) -> None:
        """
        Make edge_detected mask into image with transparent background\n
        Used in preparation for overlay of mask onto dots image
        """
        
        img = Image.open(self.outline)
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

        # convert outline and dots to png of same size
        self.tif_to_png()
        
        # get the cells outline and make transparent
        self.make_transparent()
        
        # open the image of the dots
        img1 = Image.open(self.dots)
        
        # open the image of the mask
        img2 = Image.open(self.outline)
        
        # paste the mask on top of the dots
        img1.paste(img2, (0,0), mask = img2)
        
        # save the image
        img1.save('analysis_output/overlay.png', "PNG")