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