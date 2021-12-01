# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:09:48 2021

@author: kayla

white = [255, 255, 255]

Convert png to array: https://www.kite.com/python/answers/how-to-convert-an-
                      image-to-an-array-in-python
Resize image to 360x360:https://auth0.com/blog/image-processing-in-python-
                        with-pillow/
Find neighbor pixels: https://youtu.be/RHTzPCM5vnw MonkHaus Youtube video

Potentially useful sites HERE:
    https://codestudyblog.com/cnb11/1123194848.html
    https://stackoverflow.com/questions/65930839/getting-a-centre-of-an-irregular-shape
    https://www.pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/
"""
import numpy as np
import PIL
from PIL import Image
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.data
import cv2
import itertools
import queue
from queue import SimpleQueue



def resize(image, width, height, newfile):
    """ Resize image to desired width and height
        Parameters: image, int(width), int(height), name for new image (string)
        Returns: resized image filename
    """
    image = Image.open(image)
    resized_image = image.resize((width, height))
    resized_image.save(newfile)
    return newfile

def define_border(image_array, R, G, B):
    """ Differentiate between border and non-border based on pixel color
        Parameters: image array, known RGB values of non-border color
        Returns: 1D numpy array of pixel designation string (border or non-border)
    """
    pixel_color = []
    for i in range(len(image_array)):
        if image_array[i][0] == R and image_array[i][1] == G and image_array[i][2] == B:
            pixel_color.append("non-border")
        else:
            pixel_color.append("border")
    return(np.array(pixel_color))

def reshape(image_array, width):
    """ Reshape a 1D array to desired width
        Parameters: image_array, int(width) of desired array
        Returns: reshaped array
    """
    return np.reshape(image_array, (-1, width))

def neighbor(image, index, radius, width, height):
    """ Find the neighbor pixels surrounding a pixel of interest (POI)
        Paramters:
            image: an array
            index: int representing the location of the POI
            radius: a float or int representing the radius from POI considered
            width, hieght: dimensions of the image in pixels
        Returns: a numpy array containing the pixel locations of pixels that
            neighbor the POI
    """
    width, height = image.shape
    index, radius = index, radius
    row, column = index // width, index % width
    
    # Define integer of radius (used in vertical/horizontal neighborhood calc)
    r = int(radius)
    
    # Create two neighborhood lists: vertical and horizontal
    x = np.arange(max(column - r, 0), min(column + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    
    # Create mesh grid
    X, Y = np.meshgrid(x, y)
    
    # Calculate Euclidean distances
    R = np.sqrt((X - column)**2 + (Y - row)**2)
    
    # Consider pixels within radius as neighbors of POI
    # Inlcudes vertical, horizontal, and diagonal neighbors
    # Stores data as Boolean values
    mask = R < radius
    
    # Give the position of neighboring pixels
    # 0    1 ........ 359 (row 1 image)
    # 360  361 ...... 719 (row 2 of image)
    neighbor_pixel_positions = (Y[mask] * width) + X[mask]
    return neighbor_pixel_positions

def linear_to_array(neighbor, width):
    """ Convert pixel position (linear) to pixle position (2D array) """
    row = (neighbor // width)
    column = (neighbor % width)
    return row, column

def array_to_linear(image):
    """ Convert pixel position (2D array) to pixel position (linear) """
    lst = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            current = image[i][j]
            lst.append(current)
    return np.array(lst)

def detect(linear_image, neighbors):
    q = queue.Queue()
    cell = []
    for i in range(len(neighbors)):
        pixel = linear_image[neighbors[i]]
        if pixel == "non-border":
            cell.append(pixel)
    return cell

# def cell_detection(image, radius, width, height):
#     pass
#     # instantiate a queue object
#     q = queue.Queue()
    
#     ls = []
#     for k in range(10):
#         neighbors = neighbor(image, k, radius, width, height)
#         ls.append(neighbors)
#         i, j = linear_to_array(neighbors, 360)
#         # for i, j in np.ndindex(image.shape):
#         # if image[i, j]
#     return ls, i, j
# ###
#     # linear_image = array_to_linear(image)
    
#     # for pixel in linear_image:
#     #     neighbors = neighbor(linear_image, pixel, radius, linear_image.shape[0], linear_image.shape[1])
        
#         # for i in range(len(neighbors)):
#         #     if neighbors[i] == "non-border":
#         #         q.put(neighbors[i])
        
#     # return q, neighbors

# ###
#     # for row in image:
#     #     for pixel in row:
#     #         neighbors = neighbor(image, int(pixel), radius, width, height)

#     #     # for pix in neighbors:
#     #     #     for i in range(len(pix)):
#     #     #         row, col = convert_position(pix[i], width)
#     #     #         if image[row][col] ==  "non-border":
#     #     #             q.put(pix[i])
#     # return neighbors



def main():
    # """read in png image"""
    # # image = mpimg.imread("small_test_cell_red.png")
    # # plt.imshow(image)
    # # plt.axis("off")
    
    # original = "small_test_cell_red.png"
    
    # # set desired dimensions of image to analyze
    # width, height = (360, 360)
    
    # # resize image to 360 x 360 pixels (the size of images used in project)
    # new_red_cell = resize(original, width, height, "test_cell_360x.png")
    
    # # open resized image using Pillow
    # cell_image = PIL.Image.open(new_red_cell)
    
    # # convert image to numpy array
    # image_sequence = cell_image.getdata()
    # image_array = np.array(image_sequence)
    
    # """
    # Create a list of the designation of each pixel: border or non-border
    # e.g if a pixel is white, then it is "non-border", if a pixel is any color
    # other than white, then it is "border"
    # Note to us: could only be used with cell outlines, not an overlay of
    # outlines on puncta/dots (unless we know the RGB values of background, cell
    #                          outlines, and dots... then we'd have to reajust
    #                          this function)
    # """
    # colors = define_border(image_array, 255, 255, 255)
    
    # # reshape array to mirror the dimensions of image (360 x 360)
    # # to see the array: go to 150, 100
    # reshaped_colors = reshape(colors, 360)
    
    # # make copy of array to start painting
    # image_copy = np.copy(reshaped_colors)
    
    # # Get the neighbor pixels of a pixel of interest
    # # This example finds the pixels neighboring the first pixel at location 0
    # neighbor_pixel_loc = neighbor(image_copy, 0, 1.5, image_copy.shape[0], image_copy.shape[1])
    # print(neighbor_pixel_loc)
    # # print(image_copy[0][1])
    
    # new = convert_position(neighbor_pixel_loc[3], 360)
    # # print(new)
    # # print(dimensions(image_copy))
    
    # new2 = cell_detection(image_copy, 1.5, image_copy.shape[0], image_copy.shape[1])
    # print(len(new2))
    
    """
    NOTE: Now that we have the color/designation of each pixel as border vs
    non-border, we can work on making a paint/fill algorithm that defines
    the pixels that belong to each cell in an image.
    
    I've used a simple image of a single red cell for this code, but we can
    create more complex sample images before we try the actual cell outlines
    
    The function Neighbor can be used to find neighboring pixels.
    Turn into a class to be used or a method in a class.
    """

if __name__ == "__main__":
    # main()
    """read in png image"""
    # image = mpimg.imread("small_test_cell_red.png")
    # plt.imshow(image)
    # plt.axis("off")
    
    original = "small_test_cell_red.png"
    
    # set desired dimensions of image to analyze
    width, height = (360, 360)
    
    # resize image to 360 x 360 pixels (the size of images used in project)
    new_red_cell = resize(original, width, height, "test_cell_360x.png")
    
    # open resized image using Pillow
    cell_image = PIL.Image.open(new_red_cell)
    
    # convert image to numpy array
    image_sequence = cell_image.getdata()
    image_array = np.array(image_sequence)
    
    """
    Create a list of the designation of each pixel: border or non-border
    e.g if a pixel is white, then it is "non-border", if a pixel is any color
    other than white, then it is "border"
    Note to us: could only be used with cell outlines, not an overlay of
    outlines on puncta/dots (unless we know the RGB values of background, cell
                             outlines, and dots... then we'd have to reajust
                             this function)
    """
    colors = define_border(image_array, 255, 255, 255)
    
    # reshape array to mirror the dimensions of image (360 x 360)
    # to see the array: go to 150, 100
    reshaped_colors = reshape(colors, 360)
    
    # make copy of array to start painting
    image_copy = np.copy(reshaped_colors)
    
    # Get the neighbor pixels of a pixel of interest
    # This example finds the pixels neighboring the first pixel at location 0
    neighbor_pixel_loc = neighbor(image_copy, 0, 1.5, image_copy.shape[0], image_copy.shape[1])
    print(neighbor_pixel_loc)
    print(len(neighbor_pixel_loc))
    # print(image_copy[0][1])
    
    # new = linear_to_array(neighbor_pixel_loc[3], 360)
    # print(new)
    # print(dimensions(image_copy))
    
    for i in range(len(reshaped_colors)):
        for j in range(len(reshaped_colors[i])):
            if reshaped_colors[i][j] == 1:
                print(str(i) + ',' + str(j))

    new2 = array_to_linear(image_copy)
    # print(new2)

    # new3 = cell_detection(image_copy, 1.5, image_copy.shape[0], image_copy.shape[1])
    # print(new3)
    
    new4 = detect(new2, neighbor_pixel_loc)