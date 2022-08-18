import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('/Users/DuongHoang/Research499/Advanced_Image_Analysis/data/cutout_image.fits',memmap=True)
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
#notice that the back already dark(0) so by increasing the pixel 1 the area that has objects would >1
from copy import deepcopy
mask_object = deepcopy(segmap)#copy the segmap information into a memory instance
mask_object = mask_object+1
mask_object[np.where(mask_object>1)] =0 #masking all the objects =zero
objects_masked_data= image_data *mask_object
for_histogram = [i for i in objects_masked_data.flat if i!=0] #flat: convert 2D array into 1D array
fig=plt.figure(figsize=(8,6))
histogram, bins_edges, something = plt.hist(for_histogram, bins = 70, density=True, histtype='step')
plt.xlim([-0.03,0.03])
plt.ylim([0,95])
plt.xticks([-0.02,0.01,0.00,0.01,0.02,0.03],fontsize=18)
plt.yticks([0,20,40,60,80],fontsize=18)
plt.ylabel('Number',fontsize=16)
plt.xlabel('Pixel Values', fontsize=16)
plt.text(-0.03, 100, 'Distribution of Background Sky Pixels', color='black', fontsize=14)
plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T4_a_distribution_of_background_sky.png',
            bbox_inches='tight')
plt.show()
