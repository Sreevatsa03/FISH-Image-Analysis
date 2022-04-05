from FISH_analysis import Puncta_Analysis
from FISH_analysis import Puncta_Thresholding

# overlay cells outline onto thresholded dots
analysis = Puncta_Analysis('segmentation_output/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001_outlines.png', 'Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C3 (SOX2) thrsholded dots.tif')
analysis.overlay()

"""
The below code tests dot counting on real FISH image outputs
We were able to get this to work on fake images of cells and dots made in paint but struggled with real images
The below code is just an example of the continued analysis pipeline but is not fully functional
"""

# create thresholding objects for centroids
cells = Puncta_Thresholding('analysis_output/outline.png')
dots = Puncta_Thresholding('analysis_output/dots.png')

# get centroids of cells and dots
cell_centroids = cells.get_centroids()
dot_centroids = dots.get_centroids()

# get and save all cell centroids
analysis.refine_cell_centroids(cell_centroids)

# get and save all dot centroids
analysis.refine_dot_centroids(dot_centroids)

# save a csv file of dots per cell data for real FISH images of cells and dots
analysis.dots_per_cell()