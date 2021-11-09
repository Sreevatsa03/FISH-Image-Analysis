from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Puncta_Analysis:
    """
    Instantiate analysis of cleaned (thresholded dots) spinal cord cell image or mask of spinal cord cell image.
    Various analyses can be performed on the given image.
    
    :param image: FISH image (cleaned or mask)
    :type image: .tif
    """
    
    def __init__(self, image):
        im = mpimg.imread(image)
        plt.imshow(im)
        plt.savefig('image_plot.png')