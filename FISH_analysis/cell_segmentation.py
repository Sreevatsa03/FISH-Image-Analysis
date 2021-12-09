from czifile import CziFile
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from cellpose import models, io, utils
plt.rcParams['figure.dpi'] = 300

class Segmentation:
    """
    Instantiate segmentation of cells using cellpose developed by Carsen Stringer et al. (requires cellpose package https://cellpose.readthedocs.io/en/latest/) \
    Segmentaion includes creating masks and outlines of cells

    :param cells_img: image of cells to generate masks and outlines
    :type cells_img: .png
    """

    def __init__(self, cells_img) -> None:
        path, ext = os.path.splitext(cells_img)
        filename = str(path).split('/')
        self.path = cells_img
        self.cells = filename[-1]
        self.masks = ''
        self.outlines = ''
        
    def save(self, filepath, fig=None):
        '''
        Save the current image with no whitespace using matplotlib
        
        :param str filepath: file path in which to save image
        :param fig: figure to be used,
            but if None is passed in, we create a new figure
        :type fig: matplotlib Figure instance
        '''

        # set background style
        plt.style.use('dark_background')

        # determine if we need to create a new figure
        if not fig:
            fig = plt.gcf()
        
        # plot and save image
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(filepath)
        
    def make_masks(self, flow_threshold=0.9, cellprob_threshold=-5, diameter=None, model_type='cyto') -> None:
        """
        Run cellpose on cells image to obtain masks and save as a .png \ 
        Cellpose documentation: https://cellpose.readthedocs.io/en/latest/installation.html
        
        :param float flow_threshold: increase this flow threshold if cellpose is not returning as many masks as you’d expect,
            and decrease this threshold if cellpose is returning too many ill-shaped masks.
            0.1 <= flow_threshold <= 1.2     
        :param float cellprob_threshold: decrease this mask threshold if cellpose is not returning as many masks as you’d expect, 
            and increase this threshold if cellpose is returning too masks particularly from dim areas.
            -6 <= cellprob_threshold <= 6
        :param float diameter: if value is None, size value will be estimtated (pixel size)
        :param str model_type: model type to be used during cellpose segmentation,
            cytoplasm = 'cyto', nucleus = 'nuclei'
        """

        # ensure parameters are correct
        if (flow_threshold <= 0.1) or (flow_threshold >= 1.2) or (cellprob_threshold >= 6) or (cellprob_threshold <= -6):
            print('Cellprob or flow threshold is out of range. Be sure cellprob threshsold is within (-6, 6) and flow threshold is within (0.1, 1.1)')
        files = [self.path]
        
        # show image to be segmented
        img = cv2.imread(files[-1])
        cv2.imshow("Watershed", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        # run cellpose
        model = models.Cellpose(gpu=True, model_type=model_type) # add more model types in future, cytoplasm = 'cyto', nuclei = 'nuclei'
        channels = [[2,3], [0,0], [0,0]]
        for chan, filename in zip(channels, files):
            img = io.imread(filename)
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=chan, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

            # save results so you can load in gui
            io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)
        
        # save masks
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(masks, aspect='auto', cmap = cm.Greys.reversed())
        self.save(f'segmentation_output/{self.cells}_masks.png')    
        self.masks = f'segmentation_output/{self.cells}_masks.png'
        print("Saved Mask PNG")
        cv2.imshow("Mask", self.masks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_masks(self) -> str:
        """
        Get masks path

        :return str self.masks: file path of mask
        """

        if self.masks != '':
            return self.masks
        else:
            print('Cells have not been segmented. Run Segmentation.make_masks(self, flow_threshold, cellprob_threshold, diameter, model_type)')
        
    def make_outlines(self) -> None:
        """
        Creates outlines for cells and saves as a .png \ 
        Outline can be used for analysis using Puncta_Analysis functionality
        """

        npy = os.path.splitext(self.cells)[0] + '_seg.npy'

        dat = np.load(npy, allow_pickle=True).item()
        fig = plt.figure(figsize=(5, 5), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot image with outlines overlaid in red
        outlines = utils.outlines_list(dat['masks'])
        for o in outlines:
            plt.plot(o[:,0], o[:,1], color='r')

        # save outlines
        plt.axis('off')
        fig.savefig(f'segmentation_output/{self.cells}_outlines.png') 
        self.save(f'segmentation_output/{self.cells}_outlines.png')    
        self.outlines = f'segmentation_output/{self.cells}_outlines.png'
        print("Saved Outlines PNG")
        cv2.imshow("Mask", self.outlines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class CZI_Channels:
    """ 
    Characterize a .czi file with a .png image of each channel

    :param CZI: output from FISH experiment to be characterized
    :type CZI: .czi
    """
    
    def __init__(self, Czi) -> None:
        # read czi data as array
        with CziFile(Czi) as czi:
            image_arrays = czi.asarray()
        
        # store filename and directory
        path, ext = os.path.splitext(Czi)
        filename = path.split('/')
        self.cziname = filename[-1]
        
        # shape of czi
        shape = list(image_arrays.shape)

        # store number of channels
        self.nchan = int(shape[2])
        
        # if zstack determine middle plane
        nzstack = int(shape[3])
        midzstack = round(nzstack / 2)
        
        # store color maps and array channels
        self.cmaps = [cm.Reds, cm.Greens, cm.RdPu, cm.Greys] # ADD MORE CMAPS FOR MORE CHANNELS
        self.arrchannels = [image_arrays[0, 0, idx, midzstack].T[0] for idx in range(self.nchan)]
        
    def num_channels(self) -> list[int]:
        """
        Get number of channels in czi

        :return num_chans: number of channels in czi
        :rtype num_chans: list[int]
        """

        num_chans = list(range(1, self.nchan + 1))
        return num_chans
        
    def show_all_channels(self) -> None:
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
        
        # plot image
        fig, ax_grid = plt.subplots(N_rows, N_cols, figsize=(N_cols*10,N_rows*10))
        counter = 0
        for row in range(N_rows):
            for col in range(N_cols):
                image = self.arrchannels[counter]
                ax_grid[row][col].imshow(image, cmap = self.cmaps[counter].reversed())  
                counter += 1
        
        plt.savefig(f'segmentation_output/C_All_{self.cziname}.png')
        cv2.imshow("Channels", f'segmentation_output/C_All_{self.cziname}.png')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def channel_to_png(self, C) -> None:
        """
        Save channel as .png
        
        :param int C: channel number to be saved as a .png
        """

        # ensure correct channels are being chosen
        if C > self.nchan or C <= 0:
            return print(f"Channel out of range. Select from 1 - {self.nchan}")
        else:
            # generate image and save as png
            C -= 1
            chan = self.arrchannels[C]
            img = Image.fromarray(chan)
            fig = plt.figure(figsize=(5, 5), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img, aspect='auto', cmap = self.cmaps[C].reversed())
            fig.savefig(f'segmentation_output/C{C + 1}_{self.cziname}.png')   
            cv2.imshow(f'C{C + 1}', f'segmentation_output/C{C + 1}_{self.cziname}.png')
            cv2.waitKey(0)
            cv2.destroyAllWindows()