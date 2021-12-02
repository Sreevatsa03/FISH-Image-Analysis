# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:09:48 2021

@author: kayla

Convert png to array: https://www.kite.com/python/answers/how-to-convert-an-
                      image-to-an-array-in-python
Resize image to 360x360:https://auth0.com/blog/image-processing-in-python-
                        with-pillow/
Find neighbor pixels: https://youtu.be/RHTzPCM5vnw MonkHaus Youtube video
Referenced source code for flood_fill
https://thispointer.com/find-the-index-of-a-value-in-numpy-array/
Export df as csv: https://datatofish.com/export-dataframe-to-csv/

Potentially useful sites HERE:
    https://codestudyblog.com/cnb11/1123194848.html
    https://stackoverflow.com/questions/65930839/getting-a-centre-of-an-irregular-shape
    https://www.pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/
"""
from fake_puncta_thresholding import Puncta_Thresholding
import numpy as np
import PIL
from PIL import Image
from skimage.morphology import flood_fill
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def explore_image(cell_png):
    """ Explore properties of an image (e.g., size, colors, etc.)
        Parameters:
        Returns:
    """
    pass

def resize(image, width, height, newfile):
    """ Resize image to desired width and height (in pixels)
        Parameters:
            image: numpy array of an image
            width, height: desired width and height of image (in pixels)
            newfile: a string, the name for the resized image
        Returns: filename of resized image
    """
    image = Image.open(image)
    resized_image = image.resize((width, height))
    resized_image.save(newfile)
    return newfile

def define_border(image_array, R, G, B):
    """ Differentiate between border and non-border based on pixel color of
            non-border pixels (all non-border must have same RGB)
        Parameters:
            image array: a numpy array of the image
            R, G, B: known RGB values of non-border color
        Returns: 1D numpy array of pixel designation
            border pixel = 1 and non-border pixel = 0
    """
    pixel_designation = []
    for i in range(len(image_array)):
        if image_array[i][0] == R and image_array[i][1] == G and image_array[i][2] == B:
            pixel_designation.append(0)
        else:
            pixel_designation.append(1)
    return(np.array(pixel_designation))

def reshape(image_array, width):
    """ Reshape a 1D array to 2D array of desired width
        Parameters:
            image_array: 1D numpy array
            width: integer representing desired width of 2D array in pixels
        Returns: the reshaped 2D numpy array
    """
    return np.reshape(image_array, (-1, width))

def flooding(image, x, y, newval):
    """ Flood the area inside of a cell with a new value
        Parameters:
            image: 2D numpy array of image
            x, y: integer indices of cell centroid's pixel location
            newval: the replacement value (integer, float, string, or bool)
        Returns: 2D numpy array where all pixels within cell of interest
            have value = newval
    """
    return flood_fill(image, (x, y), newval)

def define_cell(image, centroid_x, centroid_y, newval):
    """ Obtain indices of pixels within a defined border. Area inside border
        must be one color/value
        Parameters:
            image: 2D numpy array of image
            centroid_x, centroid_y: integer indices of cell centroid's pixel location
            newval: the replacement value (integer, float, string, or bool)
        Returns: numpy array (2 columns) of the x and y indices of each pixel
            inside a cell border (not including the border itself);
            size of the cell in pixels as an integer
    """
    # flood fill--- everything inside border is changed to newval
    flooded_cell = flooding(image, centroid_x, centroid_y, newval)
    
    # get indices of pixels inside cell border
    result = np.where(flooded_cell == newval)
    
    # combine x and y coordinates of cell into one numpy array
    cell_loc = np.array((result[0], result[1]), order = "F").T
    cell_size = cell_loc.size
    return cell_loc, cell_size

def cell_dictionary(cells):
    """ Create dictionary of cells with the dots per cell
        Parameters:
            cells: list of numpy arrays (pixels inside individual cells)
        Returns: a dictionary key--> cell #
                              value --> indices of pixel inside that cell
    """
    cell_dic = {}
    for i in range(len(cells)):
        cell_dic[i] = cells[i]            
    return cell_dic

def define_multiple_cells(cell_borders, cell_centroids, newval=2):
    """ Given the centroid of each cell in an image, determine which pixels
        belong to which cells
        Parameters:
            cell_borders: 2D numpy array where cell borders = 1 and background = 0
            cell_centroids: list of tuples containing (i,j) indices of
                pixel locations of the centroid of each cell in the image
            newval (optional): the new value to flood each cell with.
                Cannot be the same as preexisting border or background values
        Returns: a dictionary of numpy arrays, each of which contains the
            pixel locations (i,j) of pixels inside of a single cell;
            and a list of the sizes of the cells
    """
    cell_lst = []
    cell_size_lst = []
    for (i, j) in cell_centroids:
        cell_n, cell_n_size = define_cell(cell_borders, i, j, newval)
        cell_lst.append(cell_n)
        cell_size_lst.append(cell_n_size)
    cell_dic = cell_dictionary(cell_lst)
    return cell_dic, cell_size_lst
        
def dot_count(dots, cells, cell_num):
    """ Count the number of dots inside of a given cell
        Parameters:
            dots: a list of tuples containing the x, y indices of all dot centroids
            cells: dictionary of cells
                key --> cell # and value --> pixel loc inside that cell (i, j)
            cell_num: integer representing the key of a cell of interest
        Returns: dot count in given cell (int), list of pixels in that cell (i, j)
    """
    count = 0
    pixel_lst = []
    in_cell = []
    
    # get list of pixels inside cell of interest based on cell_num
    pixels = cells.get(cell_num)
    for i, j in pixels:
        pixel_lst.append((i, j))

    # count the number of dots inside cell of interest and create
    # list of pixel locations of each dot inside cell of interest
    for d in range(len(dots)):
        for p in range(len(pixel_lst)):
            if pixel_lst[p] == dots[d]:
                in_cell.append(pixel_lst[p])
                count += 1
    return count, in_cell

def dot_count_multiple_cells(dot_centroids, cell_dictionary):
    """ k
        Parameters:
            dot_centroids: a list of tuples containing (i,j) indices of
                pixel locations of the centroid of each dot in the image
            cell_dictionary: a dictionary of numpy arrays, each of which contains the
            pixel locations (i,j) of pixels inside of a single cell
        Returns: a list of integers, each represents the dot count for a single
            cell in the image, follows the order of given cell_dictionary
            i.e. index=0 in dot_count_list corresponds to cell 0 in cell_dictionary
    """
    dot_count_list = []
    dot_positions = []
    for i in range(len(cell_dictionary)):
        dots_n = dot_count(dot_centroids, cell_dictionary, i)
        dot_count_list.append(dots_n[0])
        dot_positions.append(dots_n[1])
    return dot_count_list, dot_positions

def dots_per_cell(cell_png, resized_filename, cell_centroids, dot_centroids,\
                  R, G, B, filepath, width=500, height=500):
    """ Export a csv containing mRNA dot signals per cell
        Parameters:
            cell_png: filename string of png image
            resized_filename: filename string for resized image
            cell_centroids: list of tuples containing indices for cell centroids (i,j)
            dots_centroids: list of tuples containing indices for dot centroids (i,j)
            R, G, B: known RGB values of background/non-border color
            filepath: filepath to save csv, a string, must follow format below
                "r'Path where you want to store the csvfile\Csvfilename.csv'"
            width, height (optional): desired image size (in pixels)
        Returns: a csv is created with dots per cell data
    """    
    # resize image --- larger image = better resolution but longer runtime
    resized = resize(cell_png, width, height, resized_filename)
    
    # open resized image using Pillow
    resized_image = PIL.Image.open(resized)
    
    # convert image to numpy array
    image_sequence = resized_image.getdata()
    image_array = np.array(image_sequence)
    
    # R, G, B values used to define cell borders: border = 1 non-border = 0
    cell_borders = define_border(image_array, R, G, B)
    
    # reshape array to mirror the dimensions of image (500 x 500) or user input
    reshaped_cell_borders = reshape(cell_borders, width)
    
    # create preliminary cell dictionary
    # key -> cell # and value -> pixel lindices (i,j) in that cell
    cell_dic, cell_sizes = define_multiple_cells(reshaped_cell_borders, cell_centroids)
    
    # create a list of dot counts and list of dots in each cell
    dot_count_list, dot_positions = dot_count_multiple_cells(dot_centroids, cell_dic)
    
    # save dot per cell data as a csv file
    save_dots_csv(cell_dic, dot_count_list, cell_sizes, dot_positions, filepath)

def save_dots_csv(cell_dic, dot_count_list, cell_sizes, dot_positions, filepath):
    """ Export a csv of dots per cell data
        Parameters:
            cell_number: a dictionary, key --> cell #
                                       value --> pixel locations inside each cell
            dot_counts: list of dots per cell (same order as dicitonary cell #)
            filepath: a string, "r'Path where you want to store the csvfile\Csvfilename.csv'"
        Returns: nothing, a csv file is saved to specified file path
    """
    # get the cell numbers from keys of cell_dic
    cell_number = []
    for key, val in cell_dic.items():
        cell_number.append(key)
        
    # create a df with 2 columns: cell # and # of dots
    dots_per_cell = pd.DataFrame()
    dots_per_cell["Cell #"] = cell_number
    dots_per_cell["# dots"] = dot_count_list
    dots_per_cell["Cell size (in pixels)"] = cell_sizes
    dots_per_cell["Dot (i,j) positions in reshaped image"] = dot_positions
    
    # convert df to csv and save to specified filepath
    dots_per_cell.to_csv(filepath, index = False)

def main():
    pass

if __name__ == "__main__":
    # main()
##############################################################################
    """ try with two_cell.png image --- works!
    """
    # # cell outline file being used and desired resized filename
    # cell_png = "two_cells.png"
    # resized_filename = "two_cells_360x.png"
    
    # # random dot centroid list I made by looking at the numpy array of the cell image
    # # the dot centroids are either in one cell, the other cell, or not in a cell
    # dots_list = [(7,4), (19,3), (15,9), (10, 7), (19, 16), (20, 13), (21, 15), (6,4)]
    
    # # list of tuples representing the centroids of each cell--- an output from
    # # Soumili and Joseph
    # cell_centroids = [(5,5), (20,15)]
    
    # # the color white (the background color of the test image used)
    # R, G, B = 255, 255, 255
    
    # """ reevaluate this csv saving method--- only works on Kayla's computer """
    # # where the csvfile is being saved
    # # filepath = r'FISH-Image-Analysis\analysis_output\dots_per_cell_twocell_test.csv'
    # filepath = r'C:\Users\kayla\project\FISH-Image-Analysis\analysis_output\dots_per_cell_twocell_test.csv'
    
    # # calling this function saves a csv file of dots per cell data
    # dots_per_cell(cell_png, resized_filename, cell_centroids, dots_list,\
    #               R, G, B, filepath)

##############################################################################
    """ try with multi_cell_test.png image --- works!
    """
    # cell outline file being used and desired resized filename
    cell_png = "multi_cell_test.png"
    resized_filename = "multi_cell_360x.png"

    # random dot centroid list I made by looking at the numpy array of the cell image
    # the dot centroids are either in one cell, the other cell, or not in a cell
    cell_centroids = [(12,9), (15,19), (9,34), (17,42), (29,16), (30,31)]
    dot_centroids = [(0, 0), (5,6), (9,14), (13,26), (0,22), (9,6), (9,7),\
                      (14,17), (9,32), (9,33), (9,34), (20, 41), (31,16),\
                          (31,28), (34,30), (24,31), (30, 34)]
    
    # the color white (the background color of the test image used)
    R, G, B = 255, 255, 255
    
    # where the csvfile is being saved
    filepath = r'C:\Users\kayla\project\FISH-Image-Analysis\analysis_output\dots_per_cell_multicell_test.csv'
    
    # calling this function saves a csv file of dots per cell data
    dots_per_cell(cell_png, resized_filename, cell_centroids, dot_centroids,\
                  R, G, B, filepath)
    
    img = mpimg.imread(resized_filename)
    imgplot = plt.imshow(img)
    
##############################################################################    
    """ after getting multi_cell_test.png to work, try with real cell outline,
        cell centroids, and dot centroids
        need =  cell outline from Antonio (a filename)
                cell centroids from Soumili and Joseph
                dot centorids from Soumili and Jospeh
    """
    # # cell outline file being used and desired resized filename
    # cell_png = "C:/Users/kayla/project/FISH-Image-Analysis/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001_otlines.png"
    # resized_filename = "resized_C4_SOX2.png"
    
    # """ To Do: get rid of border around cell outlines
    # """
    # # image = mpimg.imread(cell_png)
    # # plt.imshow(image)
    # # plt.axis("off")
    
    # # # random dot centroid list I made by looking at the numpy array of the cell image
    # # # the dot centroids are either in one cell, the other cell, or not in a cell
    
    # # # unsucessfully gets cell centroids :(
    # cell_centroids = Puncta_Thresholding(cell_png).get_centroids(0)
    
    # # # sucessfully gets a list of tuples representing dot centroids
    # dot_centroids = Puncta_Thresholding('C:/Users/kayla/project/FISH-Image-Analysis/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C2 (Pax6) thresholded dots.tif').get_centroids(0)
    
    # # # the color black (the background color of the test image used)???
    # R, G, B = 0, 0, 0
    
    # # # where the csvfile is being saved
    # filepath = r'C:\Users\kayla\project\FISH-Image-Analysis\analysis_output\testing_dot_count.csv'
    
    # # # calling this function saves a csv file of dots per cell data
    # dots_per_cell(cell_png, resized_filename, cell_centroids, dot_centroids,\
    #     R, G, B, filepath, width=1000, height=1000)
    
######## CUTS
    # """ create function: defines the borders of each cell given the centroid
    #     of each cell as a list of tuples
    # """
    # cell_1 = define_cell(reshaped_cell_borders, 5, 5, 2)
    # cell_2 = define_cell(reshaped_cell_borders, 20, 15, 2)
    
    # """ create function: creates list of numpy arrays (which represent
    #         the pixels inside of each cell)
    # """
    # # create a list of numpy arrays, each element in list is np array of pixels
    # # in a single cell
    # cell_lst = [cell_1, cell_2]
    
    # # create a dictionary of cells: key --> cell # and value --> pixels in cell
    # cell_dic_try = cell_dictionary(cell_lst)