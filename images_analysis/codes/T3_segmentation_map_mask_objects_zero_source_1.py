import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('..data/postage_stamp.fits',memmap=True)
#the postage_stamp is the cutout from large image
#memmap will ensure not to overload RAM incase the images are large
image_data =large_image[0].data
image_header= large_image[0].header
byte_swaped_data = image_data.byteswap().newbyteorder() #data of the image that we want to extract the source
#computing the global background of the image, above which sources will be identified in the cutout image
global_bkg =sep.Background(byte_swaped_data,bw=64,bh=64,fw=3,fh=3)
#Computing back ground subtract data
bkg= byte_swaped_data - global_bkg
from astropy.visualization import (MinMaxInterval, PercentileInterval, ZScaleInterval,
                                   SqrtStretch, AsinhStretch, LogStretch, ImageNormalize)
from astropy.convolution import Gaussian2DKernel #Define convolution filter with filter size=5
source_kernel = Gaussian2DKernel(5)
objects, segmap = sep.extract (bkg,3, err= global_bkg.globalrms, minarea=10,
                       deblend_nthresh=64, deblend_cont=0.001, filter_kernel=source_kernel.array,
                       segmentation_map=True)
from copy import deepcopy
mask_object = deepcopy(segmap)#copy the segmap information into a memory instance
print type(mask_object)
#notice that the back already dark(0) so by increasing the pixel 1 the area that has objects would >1
mask_object = mask_object+1
mask_object[np.where(mask_object>1)] =0 #masking all the objects =zero

fig, axs = plt.subplots(nrows = 1, ncols=4,figsize=[35,8])
axs1,axs2, axs3, axs4=axs
normalization = ImageNormalize(image_data, interval=PercentileInterval(99.), stretch= AsinhStretch())
#Plot Cutout Image
axs1.imshow(image_data, origin='lower', cmap='gray', norm=normalization)
axs2.imshow(segmap, origin='lower', cmap='nipy_spectral')
axs3.imshow(mask_object, origin='lower', cmap='nipy_spectral')
axs4.imshow(image_data * mask_object, origin='lower', cmap='gray', norm=normalization)

axs1.xaxis.set_visible(False)
axs1.yaxis.set_visible(False)
axs1.text(2,180,'Cutout Image', color='white', fontsize=12)

axs2.xaxis.set_visible(False)
axs2.yaxis.set_visible(False)
axs2.text(2,180,'Segmentation Map', color='white', fontsize=12)

axs3.xaxis.set_visible(False)
axs3.yaxis.set_visible(False)
axs3.text(2,180,'Mask Objects', color='black', fontsize=12)

axs4.xaxis.set_visible(False)
axs4.yaxis.set_visible(False)
axs4.text(2,180,'All the sources masked', color='white', fontsize=12)

plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T3_segmentation_map_mask_objects_zero_source_1.png',
            bbox_inches='tight')
plt.show()

# plt.imshow(segmap,origin='lower',cmap='nipy_spectral')
# plt.show()
