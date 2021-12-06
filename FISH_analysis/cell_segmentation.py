from czifile import CziFile
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import models, io, plot, utils

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
        
    def save(self, filepath, fig=None):
        '''
        Save the current image with no whitespace
        Example filepath: "myfig.png" or r"C:\myfig.pdf
        code derived from: 
        https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image" 
        '''
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        if not fig:
            fig = plt.gcf()
            
        
        plt.subplots_adjust(0,0,1,1,0,0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(filepath)
        
    def make_masks(self, flow_threshold, cellprob_threshold, diameter, model_type):
        """
        Run cellpose on cells image to obtain masks and save as png
        
        Cellpose documentation: https://cellpose.readthedocs.io/en/latest/installation.html
        
        Flow threshold: Increase this threshold if cellpose is not returning as many masks as you’d expect. 
        Similarly, decrease this threshold if cellpose is returning too many ill-shaped masks.      
        
        Mask thrsohold: Decrease this threshold if cellpose is not returning as many masks as you’d expect. 
        Similarly, increase this threshold if cellpose is returning too masks particularly from dim areas.
        
        Diameter: If = None, will be estimtated (pixel size)
        
        Model_type: Model used during cellpose segmentation. Cytoplasm = 'cyto', Nucleus = 'nuclei'
        """
        # diameter in pixels, estimated if diameter = None
        # ensure parameters are correct
        if flow_threshold <= 0.1 or flow_threshold >= 1.2 or cellprob_threshold >= 6 or cellprob_threshold <= -6:
            return print('Cellprob or flow threshold is out of range. Be sure cellprob threshsold is within (-6, 6) and flow threshold is within (0.1, 1.1)')
        files = [self.path]
        
        # show image to be segmented
        img = io.imread(files[-1])
        fig, ax = plt.subplots(figsize=(5,5))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(img, aspect = 'auto')
        plt.axis('off')
        plt.show()
 
        # run cellpose
        model = models.Cellpose(gpu=True, model_type= model_type) # add more model types in future, cytoplasm = 'cyto', nuclei = 'nuclei'
        channels = [[2,3], [0,0], [0,0]]
        for chan, filename in zip(channels, files):
            img = io.imread(filename)
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=chan, flow_threshold=flow_threshold,cellprob_threshold=cellprob_threshold)

            # save results so you can load in gui
            io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)
        
        # save masks in same directory as cells image
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(masks, aspect='auto', cmap = cm.Greys.reversed())
        self.save(f'{self.dir}/{self.cells}_masks.png')  
        plt.close(fig)    
        self.masks = f'{self.dir}/{self.cells}_masks.png'
        print("Saved Mask PNG")
        plt.show()

    
    def get_mask(self):
        """
        Get masks path as string
        """
        if self.masks != '':
            return self.masks
        
        else:
            return print('Cells have not been segmented. Run Segmentation.make_masks(self, flow_threshold, cellprob_threshold, diameter, model_type)')
        
    def make_outlines(self):
        """
        Creates outlines for cells and saves as a .png
        """
        npy = os.path.splitext(self.cells)[0] + '_seg.npy' #fix naming convention

        dat = np.load(npy, allow_pickle=True).item()
        fig = plt.figure(figsize=(5, 5), frameon=False)
        plt.style.use('dark_background')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot image with outlines overlaid in red
        outlines = utils.outlines_list(dat['masks'])
        for o in outlines:
            plt.plot(o[:,0], o[:,1], color='r')
    
        plt.axis('off')
        fig.savefig(f'{self.dir}/{self.cells}_outlines.png') 
        self.save(f'{self.dir}/{self.cells}_outlines.png')  
        plt.close(fig)    
        self.outlines = f'{self.dir}/{self.cells}_outlines.png'
        print("Saved Outlines PNG")
        plt.show()
        
    def get_dir(self):
        """
        Get directory as string
        """
        if self.dir != '':
            return self.dir
        
        else:
            return print('Cells have not been segmented. Run Segmentation.make_masks(self, flow_threshold, cellprob_threshold, diameter, model_type)')

class czi_chans:
    """ 
    Characterize a .czi file with .pngs of each channel
    """
    
    def __init__(self, Czi):
        """
        Store channels of czi in image dictionary (code derived from:
        http://schryer.github.io/python_course_material/python/python_10.html)
        """
        # read czi data as array
        with CziFile(Czi) as czi:
            image_arrays = czi.asarray()
        
        # store filename and directory
        path, ext = os.path.splitext(Czi)
        filename = path.split('/')
        self.dir = os.path.abspath(os.path.join(Czi, os.pardir))
        self.cziname = filename[-1]
        
        # store shape of czi, number of channels, and number of zstacks
        self.shape = list(image_arrays.shape)
        self.nchan = int(self.shape[2])
        self.nzstack = int(self.shape[3])
        
        # if zstack determine middle plane
        self.midzstack = round(self.nzstack/2)
        
        # store color maps and array channels
        self.cmaps = [matplotlib.cm.Reds,matplotlib.cm.Greens,matplotlib.cm.RdPu,matplotlib.cm.Greys] # ADD MORE CMAPS FOR MORE CHANNELS
        self.arrchannels = [image_arrays[0,0,idx,self.midzstack].T[0] for idx in range(self.nchan)]
        
    def nchans(self):
        """
        Return number of channels in czi as list 
        """
        return list(range(1, self.nchan + 1))
        
        
    def show_channel(self, C):
        """
        Display image from a channel
        """
        # ensure correct channels are being shown
        if C > self.nchan or C <= 0:
            return print(f"Channel out of range. Select from 1 - {self.nchan}")
        
        else:
            C -= 1
            chan = self.arrchannels[C]
            img = Image.fromarray(chan)
            fig = plt.figure(figsize=(5, 5), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            return ax.imshow(img, aspect='auto', cmap = self.cmaps[C].reversed())
        
    def show_all_chan(self):
        """
        Show all channels in czi
        """
        # generate appropriate display for channels
        if self.nchan % 2 == 0:
            N_rows = 2
            N_cols = round(self.nchan / 2)
        
        else:
            N_rows = 2
            N_cols = round(self.nchan / 2) + 1
        
        fig, ax_grid = plt.subplots(N_rows, N_cols, figsize=(N_cols*10,N_rows*10))
        counter = 0
        for row in range(N_rows):
            for col in range(N_cols):
                image = self.arrchannels[counter]
                ax_grid[row][col].imshow(image, cmap = self.cmaps[counter].reversed())  
                counter += 1
        
        return plt.show()

    def chan2png(self, C):
        """
        Save channel as .png
        """
        # ensure correct channels are being chosen
        if C > self.nchan or C <= 0:
            return print(f"Channel out of range. Select from 1 - {self.nchan}")
        
        # generate image and save as png
        else:
            C -= 1
            chan = self.arrchannels[C]
            img = Image.fromarray(chan)
            fig = plt.figure(figsize=(5, 5), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img, aspect='auto', cmap = self.cmaps[C].reversed())
            fig.savefig(f'{self.dir}/C{C + 1}_{self.cziname}.png') 
            plt.close(fig)    
            return print('Saved')