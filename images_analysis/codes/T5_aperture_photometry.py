import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('.../data/postage_stamp.fits',memmap=True)
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
#MEASURE THE LIGHT FROM AN OBJECTS WITHIN A APERTURE (CIRCLE)
from photutils import CircularAperture
from photutils import aperture_photometry

# Step1: define the positions of the objects
positions = [(i, j) for i, j in zip(objects['x'], objects['y'])]
# Step2: Place the aperatures at the position of the objects and define the radius of the aperatures
apertures = CircularAperture(positions, r=10)
# Step3: Measure the flux in each aperture
pho_table = aperture_photometry(bkg__subtracted, apertures)
# Step4: Show the apertures overlaid on top of the image
print (pho_table)
apertures.plot(color='red', lw=1.5, alpha=0.5, ls='--')

fig=plt.figure(figsize=(8,6))
axis=plt.gca()
normalization = ImageNormalize(bkg__subtracted, interval=PercentileInterval(99.), stretch= AsinhStretch())
axis.imshow(image_data,origin='lower',cmap='gray',norm=normalization)
axis.xaxis.set_visible(False)
axis.yaxis.set_visible(False)
apertures.plot(color='red', lw=1.5, alpha=0.5, ls='--')
axis.text(2,190,'Aperture Photometry of the Detected Objects', color='white', fontsize=14)
#title of the plot


plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T5_aperture_photometry.png',
            bbox_inches='tight')

plt.show()
