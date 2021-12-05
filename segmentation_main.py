from FISH_analysis import Segmentation
from FISH_analysis import czi_chans


# isolate channels from png and save 
c = '/Users/antoniovillanueva/Desktop/FISH-Image-Analysis/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.czi'
czi = czi_chans(c)
czi.show_channel(4)
print(czi.nchans())
for chan in (czi.nchans()):
    czi.chan2png(chan)
#%%
# list of number of chans 
print(czi.nchans())

# show a channel in a czi
czi.show_channel(3)



#%%
# generate mask of cells 
cells = Segmentation('/Users/antoniovillanueva/OneDrive - Northeastern University/NEU/2021/Data Science Intermediate/Project/Images/SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001/C4_SOX2_(G)._PAX6_(R)._PAX7_(FR)_40x_Spinal_Cords_Uninjured_001.png')
cells.make_masks(0.9, -5, None, 'cyto') 

      
# FLow threshold: Increase this threshold if cellpose is not returning as many masks as you’d expect. 
# Similarly, decrease this threshold if cellpose is returning too many ill-shaped masks.      

# Mask thrsohold: Decrease this threshold if cellpose is not returning as many masks as you’d expect. 
# Similarly, increase this threshold if cellpose is returning too masks particularly from dim areas.

# Diamter: If = None, will be estimtated (pixel size)
#%%
# get mask (returns path)
c  = cells.get_mask()
o = cells.make_outlines()