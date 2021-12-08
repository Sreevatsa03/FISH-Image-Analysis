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
Directory name: https://stackoverflow.com/questions/36476659/how-to-add-a-relative-
    path-in-python-to-find-image-and-other-file-with-a-short-p

Potentially useful sites HERE:
    https://codestudyblog.com/cnb11/1123194848.html
    https://stackoverflow.com/questions/65930839/getting-a-centre-of-an-irregular-shape
    https://www.pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/
"""
from puncta_thresholding import Puncta_Thresholding
import numpy as np
import PIL
from PIL import Image
from skimage.morphology import flood_fill
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import euclidean_distances
import os

def file_path_name(abs_dir, folder, relative_path, filename):
    """ Create filepath to save csv into
        Parameters:
            abs_dir: string, the directory
            folder: string, the folder in the directory
            relative_path: string, the relative path once inside the folder
            filename: string, name of csv to be saved
        Returns: string, the file path of the csv to be saved
    """
    script_dir = os.path.dirname(abs_dir)
    fol = folder
    rel_path = relative_path
    abs_file_path = os.path.join(script_dir, fol, rel_path)
    current_file ="dots_per_cell_" + filename +".csv"
    return abs_file_path+"\\"+current_file

def resize(image_array, width, height, newfile):
    """ Resize image to desired width and height in pixels
        Parameters:
            image: numpy array of an image
            width, height: desired width and height of image in pixels
            newfile: a string, the filename for the resized image
        Returns: filename of resized image
    """
    image = Image.open(image_array)
    resized_image = image.resize((width, height))
    resized_image.save(newfile)
    return newfile

def display_img(image, inputtype = "numpy", outputtype = "plt", colormap="gray"):
    """ Display numpy array or png using matplotlib
        Parameters:
            image: numpy array or png image
            inputtype (optional): "numpy" or "png"
            outputtype (optional): "numpy" or "png"
            colormap (optional): string for desired color map
        Returns:
    """
    if inputtype == "numpy":
        img = image
    elif inputtype == "png":
        img = mpimg.imread(image)
    plt.imshow(img, cmap="gray")
    
    if outputtype == "plt":
        plt.show()
    elif (outputtype == "numpy") and (inputtype == "png"):
        img = mpimg.imread(image)
        return img
    # plt.savefig("analysis_output/numpy_image.png")
    # plt.show()
    
def clean_centroids(lst):
    """ Remove duplicate centroids for same cell
        Parameters: list of (i,j) centroid index locations
        Returns: 2D numpy array of unique cell centroid i,j index locations
    """
    return np.array(list(set(lst)))
    
def euclidean_distance(array):
    """ Calculate euclidean distance
        Parameters: an 2D array of (i,j) indices sorted by either i or j
        Returns: 2D array of euclidean dist between each pair of (i,j) indices
    """
    return euclidean_distances(array,array)

def remove_non_centroids(array, euclideans, threshold = 50):
    """ Remove centroids that are too close to accurately depict separate centroids
        Parameters:
            array: 2D array of (i,j) index locations sorted by i or j
            euclideans: 2D array of euclidean dist between each pair of (i,j) indices
            threshold (optional): integer pixel distance between two centroids
                smaller threshold = less centroids eliminated
                larger threshold = more centroids eliminated
        Returns: an array of centroids that are unique to the object it represents
    """
    indices = []
    distances = []
    for i in range(len(euclideans)):
        for j in range(len(euclideans[i])):
            if (euclideans[i][j] <= threshold) and (euclideans[i][j] != 0):
                indices.append(i)
                distances.append(euclideans[i][j])

    indices = list(set(indices))
    array = np.delete(array, indices, axis=0)
    return array

def refine_cell_centroids(cell_png, width, height, threshold, newfile):
    """ Refine the cell centroids returned from Puncta_Thresholding.get_centroids()
        Parameters:
            cell_png: filepath for cell outline image
            width, height: desired dimensions of resized image in pixels
            threshold: integer from [0, 255]
                all pixel values below threshold become 0 (black)
                all pixel values above threshold become 255 (white)
            newfile: string name for resized image
        Returns: a list of cell centroid indices (i,j)
    """
    resized_cell_outline = resize(cell_png, width, height, newfile)
    cell_centroids = Puncta_Thresholding(resized_cell_outline).get_centroids(threshold)
    clean_cell_centroids = clean_centroids(cell_centroids)
    sorted_array = clean_cell_centroids[np.argsort(clean_cell_centroids[:,1])]
    sorted_euclideans = euclidean_distance(sorted_array)
    nontrivial_cells = remove_non_centroids(sorted_array, sorted_euclideans)
    return list(map(tuple, nontrivial_cells))

def refine_dot_centroids(dots_png, width, height, threshold, newfile):
    """ Refine the dot centroids returned from Puncta_Thresholding.get_centroids()
        Parameters:
            dots_png: filepath for dots image
            width, height: desired dimensions of resize image in pixels
            threshold: integer from [0, 255]
                all pixel values below threshold become 0 (black)
                all pixel values above threshold become 255 (white)
            newfile: string name for resized image
        Returns: a list of dot centroid indices (i,j)
    """
    resized_dots = resize(dots_png, width, height, newfile)
    dot_centroids = Puncta_Thresholding(resized_dots).get_centroids(threshold)
    return dot_centroids

def define_border(image_array, R, G, B):
    """ Differentiate between border and non-border based on pixel color of
            non-border pixels (all non-border pixels must have same RGB)
        Parameters:
            image_array: numpy array of the image
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
    """ Flood the area inside of a cell with a new value given its centroid
        coordinates
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
            inside a cell border (not including the border itself); and the
            size of the cell in pixels
    """
    # flood_fill--- everything inside border is changed to newval
    flooded_cell = flooding(image, centroid_x, centroid_y, newval)
    
    # get indices of pixels inside cell border
    result = np.where(flooded_cell == newval)
    
    # combine x and y coordinates of cell into one numpy array
    cell_loc = np.array((result[0], result[1]), order = "F").T
    cell_size = cell_loc.size
    return cell_loc, cell_size

def cell_dictionary(cells):
    """ Create dictionary of cell #'s with the pixels inside each cell
        Parameters:
            cells: list of numpy arrays (pixels inside individual cells)
        Returns: a dictionary key--> cell #
                              value --> indices of pixels inside that cell
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
            cell_centroids: list of tuples (i,j), the index locations of
                each cell centroid in the image
            newval (optional): the new value to flood each cell with.
                Cannot be the same as preexisting border or background values
        Returns: a dictionary of numpy arrays, each of which contains the
            pixel locations (i,j) of pixels inside of a single cell;
            and a list of the sizes of the cells (integers)
    """
    cell_lst = []
    cell_size_lst = []
    
    # for each cell centroid, define the pixels inside the cell and determine
    # its size
    for (i, j) in cell_centroids:
        cell_n, cell_n_size = define_cell(cell_borders, i, j, newval)
        cell_lst.append(cell_n)
        cell_size_lst.append(cell_n_size)
    cell_dic = cell_dictionary(cell_lst)
    return cell_dic, cell_size_lst
        
def dot_count(dots, cells, cell_num):
    """ Count the number of dots inside of a single, specified cell
        Parameters:
            dots: a list of tuples containing the (i,j) indices of all dot centroids
            cells: dictionary of cells
                key --> cell #
                value --> (i, j) indices of pixels inside that cell 
            cell_num: integer representing the key of a cell of interest
        Returns: dot count in given cell (int), list of pixels in that cell (i,j)
    """
    dot_count = 0
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
                dot_count += 1
    return dot_count, in_cell

def dot_count_multiple_cells(dot_centroids, cell_dictionary):
    """ Count the number of dots in each cell for all cells in an image
        Parameters:
            dot_centroids: a list of tuples containing the (i,j) indices of
                all dot centroids
            cell_dictionary: dictionary of cells
                key --> cell #
                value --> (i, j) indices of pixels inside that cell 
        Returns: a list of dot count per cell (ints) (follows cell# order of cell_dictionary);
            and list of (i,j) index locations of dots in each cell
    """
    dot_count_list = []
    dot_positions = []
    for i in range(len(cell_dictionary)):
        dots_n = dot_count(dot_centroids, cell_dictionary, i)
        dot_count_list.append(dots_n[0])
        dot_positions.append(dots_n[1])
    return dot_count_list, dot_positions

def save_dots_csv(cell_dic, dot_count_list, cell_sizes, dot_positions, filepath):
    """ Export a csv of dots per cell data
        Parameters:
            cell_number: a dictionary, key --> cell #
                                       value --> (i,j) indices of pixels in that cell
            dot_count_list: list of dots per cell (same order as cell_dic cell #)
            cell_sizes: list of cell sizes
            dot_positions: list of (i,j) index locations of dots inside each cell
            filepath: a string, "r'Path where you want to store the csvfile\Csvfilename.csv'"
        Returns: nothing, a csv file is saved to specified file path
    """
    # get the cell numbers from keys of cell_dic
    cell_number = []
    for key, val in cell_dic.items():
        cell_number.append(key)
        
    # create a df with 4 columns: cell #, # of dots, cell size, dot positions
    dots_per_cell = pd.DataFrame()
    dots_per_cell["Cell #"] = cell_number
    dots_per_cell["# dots"] = dot_count_list
    dots_per_cell["Cell size (in pixels)"] = cell_sizes
    dots_per_cell["Dot (i,j) positions in reshaped image"] = dot_positions
    
    # convert df to csv and save to specified filepath
    dots_per_cell.to_csv(filepath, index = False)
    
def dots_per_cell(cell_png, resized_filename, cell_centroids, dot_centroids,\
                  R, G, B, filepath, width=500, height=500):
    """ Export a csv containing dots per cell data
        Parameters:
            cell_png: filename string of png image
            resized_filename: filename string for resized image
            cell_centroids: list of (i,j) tuples, the indices of cell centroids
            dots_centroids: list of (i,j) tuples, the indices of dot centroids
            R, G, B: known RGB values of background/non-border color
            filepath: filepath to save csv, must follow format below
                "r'Path where you want to store the csvfile\Csvfilename.csv'"
            width, height (optional): desired image size in pixels
        Returns: a csv is created with dots per cell data
    """    
    # resize image --- larger image = better resolution but longer runtime
    resized = resize(cell_png, width, height, resized_filename)
    
    # open resized image using PIL and convert to numpy array
    resized_image = PIL.Image.open(resized)
    image_sequence = resized_image.getdata()
    image_array = np.array(image_sequence)
    
    # R, G, B values used to define cell border pixels: border = 1 non-border = 0
    cell_borders = define_border(image_array, R, G, B)
    
    # reshape array to mirror the dimensions of image (500 x 500) or user input
    reshaped_cell_borders = reshape(cell_borders, width)
    
    # create preliminary cell dictionary and determine size of each cell
    # key -> cell # and value -> (i,j) indices of pixels in that cell
    cell_dic, cell_sizes = define_multiple_cells(reshaped_cell_borders, cell_centroids)
    
    # create a list of dot counts and list of dots' positions in each cell
    dot_count_list, dot_positions = dot_count_multiple_cells(dot_centroids, cell_dic)
    
    # save dot per cell data as a csv file
    save_dots_csv(cell_dic, dot_count_list, cell_sizes, dot_positions, filepath)

def main():
    """
    (1) Test with two_cell.png image. Cell and dot centroid lists were manually
    created by looking at the numpy array of the cell outline image. Cell
    centroids are inside of a cell. Dot centroids are either inside one
    of the cells or not in any cell. 
    """
    # Cell outline file being used and desired resized filename
    cell_png = "two_cells.png"
    resized_filename = "two_cells_resized.png"
    
    # Dot centroid list manually created by looking at the numpy array of
    # the cell image. Dots are either in one cell or not in any cell
    # List of tuples representing the centroids of each cell
    cell_centroids = [(5,5), (20,15)]
    dot_centroids = [(7,4), (19,3), (15,9), (10, 7), (19, 16), (20, 13), (21, 15), (6,4)]
    
    # RGB of background color of the image used (white)
    R, G, B = 255, 255, 255
    
    # where the csvfile is saved
    filepath = file_path_name(r'C:\Users\kayla\project\FISH-Image-Analysis', "FISH-Image-Analysis", "analysis_output", "two_cells")
    
    # save a csv file of dots per cell data for two_cells.png
    dots_per_cell(cell_png, resized_filename, cell_centroids, dot_centroids,\
                  R, G, B, filepath)

    """
    (2) Test with multi_cell_test.png image. Cell and dot centroid lists
    generated as above test example with two_cell.png (1)
    """
    # Cell outline file being used and desired resized filename
    multi_cell_png = "multi_cell_test.png"
    resized_filename_multi = "multi_cell_resized.png"

    # Dot centroid list manually created by looking at the numpy array of
    # the cell image. Dots are either in one cell or not in any cell.
    # List of tuples representing the centroids of each cell
    multi_cell_centroids = [(12,9), (15,19), (9,34), (17,42), (29,16), (30,31)]
    multi_dot_centroids = [(0, 0), (5,6), (9,14), (13,26), (0,22), (9,6), (9,7),\
                      (14,17), (9,32), (9,33), (9,34), (20, 41), (31,16),\
                          (31,28), (34,30), (24,31), (30, 34)]
    
    # where the csvfile is saved
    filepath_multi = file_path_name(r'C:\Users\kayla\project\FISH-Image-Analysis', "FISH-Image-Analysis", "analysis_output", "multiple_cells")
    
    # save a csv file of dots per cell data for multi_cell_test.png
    dots_per_cell(multi_cell_png, resized_filename_multi, multi_cell_centroids,\
                  multi_dot_centroids, R, G, B, filepath_multi)
    
    """ 
    (3) Attempt to use real cell outlines (from cell_segmentation)
    and cell and dot centroids (from Puncta_Thresholding). Cell and dot
    centroids generated using get_centroids method from Puncta_Thresholding
    """
    # cell outline image from cell_segmentation, desired resized filename,
    # and thresholded dots image from puncta thresholding
    real_cell_png = "C:/Users/kayla/project/FISH-Image-Analysis/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001_otlines.png"
    resized_filename_real = "resized_cell_outline.png"
    real_dots_png = 'C:/Users/kayla/project/FISH-Image-Analysis/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C2 (Pax6) thresholded dots.tif'
    
    # reszie cell outline image, then get all cell centroids
    real_cell_centroids = refine_cell_centroids(real_cell_png, 1500,1500, 0, "resized_cell_outline.png")
    
    # resize thresholded dots image, then get all dot centroids
    real_dot_centroids = refine_dot_centroids(real_dots_png, 1500, 1500, 0, "resized_thresholded_dots.png")
    
    # RGB of background color of the image used (black)
    R2, G2, B2 = 0, 0, 0
    
    # where the csvfile is saved
    filepath_real = file_path_name(r'C:\Users\kayla\project\FISH-Image-Analysis', "FISH-Image-Analysis", "analysis_output", "real_FISH_images")
    
    # save a csv file of dots per cell data for real FISH images of cells and dots
    # cell_dict, cell_sizez = dots_per_cell(real_cell_png, resized_filename_real,\
    #                                       real_cell_centroids, real_dot_centroids,\
    #                                       R2, G2, B2, filepath_real,\
    #                                       width=1500, height=1500)

if __name__ == "__main__":
    main()