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