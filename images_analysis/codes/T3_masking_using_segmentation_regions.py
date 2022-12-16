import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('.../data/postage_stamp.fits',memmap=True)
#the postage_stamp is the cutout from large image
#memmap will ensure not to overload RAM incase the images are large
image_data =large_image[0].data
image_header= large_image[0].header
byte_swaped_data = image_data.byteswap().newbyteorder() #data of the image that we want to extract the source
#computing the global background of the image, above which sources will be identified in the cutout image
global_bkg =sep.Background(byte_swaped_data,bw=64,bh=64,fw=3,fh=3)
#Computing back ground subtract data
bkg__subtracted = byte_swaped_data - global_bkg
from astropy.visualization import (MinMaxInterval, PercentileInterval, ZScaleInterval,
                                   SqrtStretch, AsinhStretch, LogStretch, ImageNormalize)
from astropy.convolution import Gaussian2DKernel #Define convolution filter with filter size=5
source_kernel = Gaussian2DKernel(5)
objects, segmap = sep.extract (bkg__subtracted,3, err= global_bkg.globalrms, minarea=10,
                       deblend_nthresh=64, deblend_cont=0.001, filter_kernel=source_kernel.array,
                       segmentation_map=True)

from copy import deepcopy
mask = deepcopy(segmap) #copy the segmap information into a memory instance
mask[np.where(mask== 2)] = 0 #mask the second objects labelled in segmap
masked_data = mask*segmap #store the data to plot the segmentation map and the 2nd source masked
masked_data1 = mask*image_data #store the data to plot the cutout image and the 2nd source masked
fig, axs = plt.subplots(nrows = 1, ncols=4,figsize=[42,10])
axis=plt.gca()
axs1,axs2, axs3, axs4=axs
normalization = ImageNormalize(bkg__subtracted, interval=PercentileInterval(99.), stretch= AsinhStretch())
#Plot Cutout Image
axs1.imshow(image_data,origin='lower', cmap='gray', norm=normalization)
axs1.xaxis.set_visible(False)
axs1.yaxis.set_visible(False)
axs1.text(50,190,'Cutout Image', color='white', fontsize=18)
#Plot segmentation map
axs2.imshow(segmap, origin='lower', cmap='nipy_spectral')
axs2.xaxis.set_visible(False)
axs2.yaxis.set_visible(False)
axs2.text(50,190,'Segmentation Map', color='white', fontsize=18)
#Plot the segmentation map with the 2nd source masked
axs3.imshow(masked_data, origin='lower', cmap='nipy_spectral')
axs3.xaxis.set_visible(False)
axs3.yaxis.set_visible(False)
axs3.text(2,190,'Segmentation Map with the 2nd source masked', color='white', fontsize=16)
#Plot the image with the 2nd source masked
axs4.imshow(masked_data1, origin='lower',cmap='gray',norm=normalization)
axs4.xaxis.set_visible(False)
axs4.yaxis.set_visible(False)
axs4.text(7,190,'Cutout image with the 2nd source masked', color='white', fontsize=16)


plt.savefig('.../plots/T3_multipanel_masking_using_segmentation.png',
            bbox_inches='tight')
plt.show()
