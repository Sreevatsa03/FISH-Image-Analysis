import czifile as cz # You will first have to install this library by typing “pip install czifile”
from czifile import CziFile
import os
from czifile import czi2tif
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm



class czi_chans:
    """ 
    Characterize a .czi file with .pngs of each channel
    """
    
    def __init__(self, Czi):
        """
        Store channels of czi in image dictionary (code derived from :
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
