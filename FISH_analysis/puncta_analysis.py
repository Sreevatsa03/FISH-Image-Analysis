from PIL import Image
import numpy as np
from skimage.morphology import flood_fill
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import euclidean_distances
from typing import Any


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
        self.outline
        self.cell_centroids
        self.dot_centroids
    
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
        img.save('analysis_output/outline.png', "PNG")
        self.outline = 'analysis_output/outline.png'
        
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

    def remove_non_centroids(self, array, euclideans, threshold=50) -> Any:
        """ 
        Remove centroids that are too close to accurately depict separate centroids
        
        :param array: 2D array of (i,j) index locations sorted by i or j
        :param euclideans: 2D array of euclidean dist between each pair of (i,j) indices
        :param float threshold: minimum pixel distance between centroids to use as threshold for real centroids,
            and smaller threshold = less centroids eliminated, \ 
            and larger threshold = more centroids eliminated
        :type array: two dimensional numpy array
        :type euclideans: two dimensional numpy array
        :return array: array of centroids that are unique to the object it represents
        :rtype array: two dimensional numpy array
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
    
    def refine_cell_centroids(self, centroids) -> list[tuple[int, int]]:
        """ 
        Refine and save the list cell centroids
        
        :param centroids: estimated centers of cells,
            which can be made from Puncta_Thresholding.get_centroids()
        :type centroids: list[tuple[int, int]]
        """

        clean_cell_centroids = np.array(list(set(centroids)))
        sorted_array = clean_cell_centroids[np.argsort(clean_cell_centroids[:,1])]
        sorted_euclideans = euclidean_distances(sorted_array, sorted_array)
        nontrivial_cells = self.remove_non_centroids(sorted_array, sorted_euclideans)
        refined_centroids = list(map(tuple, nontrivial_cells))
        self.cell_centroids = refined_centroids

    def refine_dot_centroids(self, centroids) -> list[tuple[int, int]]:
        """ 
        Refine and save the list dot centroids
        
        :param centroids: estimated centers of cells,
            which can be made from Puncta_Thresholding.get_centroids()
        :type centroids: list[tuple[int, int]]
        """

        clean_dot_centroids = np.array(list(set(centroids)))
        sorted_euclideans = euclidean_distances(clean_dot_centroids, clean_dot_centroids)
        nontrivial_cells = self.remove_non_centroids(clean_dot_centroids, sorted_euclideans, 5)
        refined_centroids = list(map(tuple, nontrivial_cells))
        self.dot_centroids = refined_centroids

    def define_border(self, image_array, R, G, B) -> Any:
        """ 
        Differentiate between border and non-border based on pixel color of non-border pixels (all non-border pixels must have same RGB)
        
        :param image_array: numpy array of the image
        :param int R, G, B: known RGB values of non-border color
        :return border: array of pixel designation,
            and border pixel = 1,
            and non-border pixel = 0
        :rtype border: one dimensional numpy array
        """

        pixel_designation = []
        for i in range(len(image_array)):
            if image_array[i][0] == R and image_array[i][1] == G and image_array[i][2] == B:
                pixel_designation.append(0)
            else:
                pixel_designation.append(1)
        border = (np.array(pixel_designation))
        return border
    
    def define_cell(self, image, centroid_x, centroid_y, newval) -> tuple[Any, int]:
        """ 
        Obtain indices of pixels within a defined border\ 
        Area inside border must be one color value
        
        :param image: array of image
        :param int centroid_x, centroid_y: indices of cell centroid's pixel location
        :param int newval: replacement value
        :type image: numpy array
        :return cell_loc: array the x and y indices of each pixel inside a cell border (not including the border itself)
        :return int cell_size: size of the cell in pixels
        :rtype cell_loc: two dimensional numpy array
        """

        # flood_fill--- everything inside border is changed to newval
        flooded_cell = flood_fill(image, (centroid_x, centroid_y), newval)
        
        # get indices of pixels inside cell border
        result = np.where(flooded_cell == newval)
        
        # combine x and y coordinates of cell into one numpy array
        cell_loc = np.array((result[0], result[1]), order = "F").T
        cell_size = cell_loc.size
        return cell_loc, cell_size

    def cell_dictionary(self, cells) -> dict:
        """ 
        Create dictionary of cell #'s with the pixels inside each cell
        
        :param cells: lists of pixels inside individual cells
        :type cells: list[ndarray]
        :return cell_dict: dictionary of cells and their pixels,
            and key --> cell #,
            and value --> indices of pixels inside that cell
        """

        cell_dict = {}
        for i in range(len(cells)):
            cell_dict[i] = cells[i]            
        return cell_dict

    def define_multiple_cells(self, cell_borders, cell_centroids, newval=2) -> tuple[dict, list[int]]:
        """ 
        Given the centroid of each cell in an image, determine which pixels belong to which cells
        
        :param cell_borders: array where cell borders = 1 and background = 0
        :param cell_centroids: index locations of each cell centroid in the image
        :param int newval: new value to flood each cell with
        :type cell_borders: two-dimensional numpy array
        :type cell_centroids: list[tuple[int, int]]
        :return cell_dict: a dictionary of arrays, 
            each of which contains the pixel locations (i,j) of pixels inside of a single cell
        :return cell_size_lst: list of the sizes of the cells
        :rtype cell_dict: dict
        :rtype cell_size_lst: list[int]
        """

        cell_lst = []
        cell_size_lst = []
        
        # for each cell centroid, define the pixels inside the cell and determine
        # its size
        for (i, j) in cell_centroids:
            cell_n, cell_n_size = self.define_cell(cell_borders, i, j, newval)
            cell_lst.append(cell_n)
            cell_size_lst.append(cell_n_size)
        cell_dict = self.cell_dictionary(cell_lst)
        return cell_dict, cell_size_lst

    def dot_count(self, dots, cells, cell_num) -> tuple[int, list[int, int]]:
        """ 
        Count the number of dots inside of a single, specified cell
        :param dots: a list containing the (i,j) indices of all dot centroids
        :param dict cells: dictionary of cells,
            and key --> cell #,
            and value --> (i, j) indices of pixels inside that cell 
        :param int cell_num: key of a cell of interest
        :type dots: list[int]
        :return int dot_count: dot count in given cell
        :return in_cell: list of pixels in that cell (i,j)
        :rtype in_cell: list[int]
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

    def dot_count_multiple_cells(self, cell_dictionary) -> tuple[list[int], list[tuple[int, int]]]:
        """ 
        Count the number of dots in each cell for all cells in an image
        :param dict cell_dictionary: dictionary of cells,
            and key --> cell #,
            and value --> (i, j) indices of pixels inside that cell 
        :return dot_count_list: list of dot count per cell (follows cell# order of cell_dictionary)
        :return dot_positions: list of (i,j) index locations of dots in each cell
        :rtype dot_count_list: list[int]
        :rtype dot_positions: list[tuple[int, int]]
        """

        dot_count_list = []
        dot_positions = []
        for i in range(len(cell_dictionary)):
            dots_n = self.dot_count(self.dot_centroids, cell_dictionary, i)
            dot_count_list.append(dots_n[0])
            dot_positions.append(dots_n[1])
        return dot_count_list, dot_positions

    def save_dots_csv(self, cell_dict, cell_sizes, dot_count_list, dot_positions) -> None:
        """ 
        Export a csv of dots per cell data

        :param dict cell_dictionary: dictionary of cells,
            and key --> cell #,
            and value --> (i, j) indices of pixels inside that cell
        :param cell_sizes: list of cell sizes
        :param dot_count_list: list of dot count per cell (follows cell# order of cell_dictionary)
        :param dot_positions: list of (i,j) index locations of dots inside each cell
        :type cell_sizes: list[int]
        :type dot_count_list: list[int]
        :type dot_positions: list[tuple[int, int]]
        """

        # get the cell numbers from keys of cell_dict
        cell_number = []
        for key, val in cell_dict.items():
            cell_number.append(key)
        
        # create a df with 4 columns: cell #, # of dots, cell size, dot positions
        dots_per_cell = pd.DataFrame()
        dots_per_cell["Cell #"] = cell_number
        dots_per_cell["# dots"] = dot_count_list
        dots_per_cell["Cell size (in pixels)"] = cell_sizes
        dots_per_cell["Dot (i,j) positions in reshaped image"] = dot_positions
        
        # convert df to csv and save to specified filepath
        dots_per_cell.to_csv('analysis_output/dots_per_cell.csv', index = False)
    
    def dots_per_cell(self) -> None:
        """ 
        Export a csv containing dots per cell data

        :param cell_centroids: list of (i,j) tuples, the indices of cell centroids
        :param dots_centroids: list of (i,j) tuples, the indices of dot centroids
        :type cell_centroids: list[tuple[int, int]]
        :type dot_centroids: list[tuple[int, int]]
        """    
        
        # open image using PIL and convert to numpy array
        image = Image.open(self.outline)
        image_sequence = image.getdata()
        image_array = np.array(image_sequence)
        
        # R, G, B values used to define cell border pixels: border = 1 non-border = 0
        cell_borders = self.define_border(image_array, 255, 255, 255)
        
        # create preliminary cell dictionary and determine size of each cell
        # key -> cell # and value -> (i,j) indices of pixels in that cell
        cell_dict, cell_sizes = self.define_multiple_cells(cell_borders, self.cell_centroids)
        
        # create a list of dot counts and list of dots' positions in each cell
        dot_count_list, dot_positions = self.dot_count_multiple_cells(self.dot_centroids, cell_dict)
        
        # save dot per cell data as a csv file
        self.save_dots_csv(cell_dict, dot_count_list, cell_sizes, dot_positions)