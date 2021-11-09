from PIL import Image


class Puncta_Analysis:
    """
    Instantiate analysis of cleaned (thresholded dots) spinal cord cell image or mask of spinal cord cell image.
    Various analyses can be performed on the given image.
    
    :param image: FISH image (cleaned or mask)
    :type image: .tif
    """
    
    def __init__(self, image):
        im = Image.open(image)