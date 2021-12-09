from FISH_analysis import Segmentation
from FISH_analysis import CZI_Channels

# isolate channels from png and save
c = 'Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.czi'
czi = CZI_Channels(c)

# # convert all channels of czi to png
# for chan in (czi.num_channels()):
#     czi.channel_to_png(chan)

# # list of number of channels
# print(czi.num_channels())

# show a channel in a czi
czi.show_channel(3)

# generate mask of cells 
cells = Segmentation('segmentation_output/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.png')

# make masks and outlines
cells.make_masks(0.9, -5, None, 'cyto') 
cells.make_outlines()