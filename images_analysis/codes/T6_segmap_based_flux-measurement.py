import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('/Users/DuongHoang/Research499/Advanced_Image_Analysis/codes/postage_stamp.fits',memmap=True)
#memmap will ensure not to overload RAM incase the images are large
image_data =large_image[0].data
image_header= large_image[0].header
#data of the image that we want to extract the source
byte_swaped_data = image_data.byteswap().newbyteorder()
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
mask = deepcopy(segmap)
mask[np.where(mask!=2)] = 0 #everything else is zero except object 2
mask[np.where(mask>1)]=1 #the pixel values in segmentation map can range between 1 and above so anything larger than 1 =1
flux = np.sum(mask*image_data)
print (flux)
fig, axs = plt.subplots(nrows = 1, ncols=2,figsize=[8,6])
axs1,axs2=axs
axs1.imshow(mask,origin='lower',cmap='PuBu')
axs1.xaxis.set_visible(False)
axs1.yaxis.set_visible(False)
axs1.text(2,180,'Object 2', color='black', fontsize=12)

axs2.imshow(mask*image_data,origin='lower',cmap='PuBu')
axs2.xaxis.set_visible(False)
axs2.yaxis.set_visible(False)
axs2.text(2,180,'Object 2 in the image', color='black', fontsize=12)
plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T6_Segmap_based_flux_measurement.png',
            bbox_inches='tight')
plt.show()
