from FISH_analysis import Puncta_Thresholding

thresholding = Puncta_Thresholding('C2 (Pax6) thresholded dots.tif')
thresholding.get_centroids(0)

thresholding2 = Puncta_Thresholding('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/MAX_C3-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif')
thresholding2.get_centroids(200)
thresholding2.binary_threshold(190)
thresholding2.erosion(100)

thresholding3 = Puncta_Thresholding('Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/Input/MAX_C4-SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.tif')
thresholding3.watershed("plot")
thresholding3.gaussian_blur(11)