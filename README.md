# FISH Analysis
A library with the funcionality to prepare and analyze HCR-FISH images. Using this library, you can construct a basic FISH image anlysis pipeline.

## Installation
```
pip install FISH_analysis
```

## Get started

### Cell Segmentation
- Separate the color channels in a czi and save as images
- Create mask and outline of cells in FISH image

```Python
from FISH_analysis import Segmentation
from FISH_analysis import CZI_Channels

# isolate channels from png and save
czi = CZI_Channels('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.czi')

# convert all channels of czi to png
for chan in (czi.num_channels()):
    czi.channel_to_png(chan)

# list of number of channels
print(czi.num_channels())

# show image of all channels in czi
czi.show_all_channels()

# segmentation of cells -> mask and outline 
cells = Segmentation('segmentation_output/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.png')

# make masks and outlines
cells.make_masks(0.9, -5, None, 'cyto') 
cells.make_outlines()
```

### Puncta Thresholding
- Threshold FISH image to isolate puncta (dots) and get rid of noise
- Get centroids of objects (cells or dots) in given image (cells outline or thresholded dots)

```Python
from FISH_analysis import Puncta_Thresholding

# centroids
thresholding = Puncta_Thresholding('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C2 (Pax6) thresholded dots.tif')
centroids = thresholding.get_centroids(0)
print(len(centroids))

# binary threshold and erosion/dilation
thresholding2 = Puncta_Thresholding('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/MAX_C3-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif')
thresholding2.binary_threshold(190, 'plot')
thresholding2.erosion_dilation(100)

# watershed and gaussian blur
thresholding3 = Puncta_Thresholding('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/MAX_C4-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif')
thresholding3.watershed("plot")
thresholding3.gaussian_blur(11, "plot")
```

### Puncta Analysis
- Threshold FISH image to isolate puncta (dots) and get rid of noise
- Get centroids of objects (cells or dots) in given image (cells outline or thresholded dots)

```Python
from FISH_analysis import Puncta_Analysis
from FISH_analysis import Puncta_Thresholding

# overlay cells outline onto thresholded dots
analysis = Puncta_Analysis('segmentation_output/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001_outlines.png', 'Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/C3 (SOX2) thrsholded dots.tif')
analysis.overlay()

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
```