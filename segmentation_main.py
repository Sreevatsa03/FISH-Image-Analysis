import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import models, io, plot, utils
import os
import matplotlib.image as mpimg

class Segmentation:
    """
    Segmentation of cells using cellpose developed by Carsen Stringer et al. (requires cellpose conda environment https://cellpose.readthedocs.io/en/latest/)
    Mask can be saved in directory of cells image

    :cells_img: .png of cells to generate mask
    :type cells_img: path of .png
    """
    def __init__(self, cells_img):
        """
        Store path, directory, and filenames of cell/masks as strings
        """
        # store path to cell image
        path, ext = os.path.splitext(cells_img)
        filename = path.split('/')
        self.path = cells_img
        
        # store directory of cell image
        self.dir = os.path.abspath(os.path.join(cells_img, os.pardir))
        
        # store filename of cell image
        self.cells = filename[-1]
        
        # initialize masks path
        self.masks = ''

        # initialize mask outlines
        self.outlines = ''
        
    def make_masks(self, flow_threshold, cellprob_threshold, diameter):
        """
        Run cellpose on cells image to obtain masks and save as png
        """
        # daimater in pixels, estimated if diameter = None
        # ensure parameters are correct
        if flow_threshold <= 0.1 or flow_threshold >= 1.2 or cellprob_threshold >= 6 or cellprob_threshold <= -6:
            return print('Cellprob or flow threshold is out of range. Be sure cellprob threshsold is within (-6, 6) and flow threshold is within (0.1, 1.1)')
        files = [self.path]
        
        # show image to be segmented
        img = io.imread(files[-1])
        plt.figure(figsize=(2,2))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        # run cellpose on image code from Carsen Stringer and colab #CITE
        model = models.Cellpose(gpu=True, model_type='cyto') # add more model types in future
        channels = [[2,3], [0,0], [0,0]]
        for chan, filename in zip(channels, files):
            img = io.imread(filename)
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=chan, flow_threshold=flow_threshold,cellprob_threshold=cellprob_threshold)

            # save results so you can load in gui
            io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)

            # save results as png
            # io.save_to_png(img, masks, flows, filename)
        
        # save masks in same directory as cells iamge
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(masks, aspect='auto', cmap = cm.Greys.reversed())
        fig.savefig(f'{self.dir}/{self.cells}_masks.png')  
        plt.close(fig)    
        self.masks = f'{self.dir}/{self.cells}_masks.png'
        print("Saved Mask PNG")
        return plt.show()
    
    def get_mask(self):
        """
        Get masks path as string
        """
        if self.masks != '':
            return self.masks
        
        else:
            return print('Cells have not been segmented. Run Segmentation.make_masks(self, flow_threshold, cellprob_threshold, diameter)')
        
    def make_outlines(self):
        """
        Create a .npy file for outlines and save outlines overlayed to image as .png
        """
        # Get numpy file path and load the data
        npy = os.path.splitext(self.path)[0] + '_seg.npy'
        dat = np.load(npy, allow_pickle=True).item()

        # generate image
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(dat['img'], aspect='auto', cmap = cm.Greys.reversed())
        
        # plot image with outlines overlaid in red (code derived from cellpose.outputs)
        outlines = utils.outlines_list(dat['masks'])
        plt.imshow(dat['img'])
        for o in outlines:
            plt.plot(o[:,0], o[:,1], color='r')
    
        # save outlines in same directory as cells image
        fig.savefig(f'{self.dir}/{self.cells}_otlines.png')  
        plt.close(fig)    
        self.outlines = f'{self.dir}/{self.cells}_outlines.png'
        print("Saved Outlines PNG")
        return plt.show()

    def get_outlines(self):
        """
        Get outlines path as string
        """
        if self.outlines != '':
            return self.outlines
        
        else:
            return print('Cells have not been segmented. Run Segmentation.make_masks(self, flow_threshold, cellprob_threshold, diameter), them make_outlines() on Segementation object.')
