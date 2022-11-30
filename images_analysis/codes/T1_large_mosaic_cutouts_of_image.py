import matplotlib.pyplot as plt
from astropy.io import fits

large_image= fits.open('.../data/large_mosaic.fits', memmap=True)
large_image_data = large_image[0].data
large_image_header=large_image[0].header

from astropy.wcs import WCS #world coordinate system to map the pixels in the image to the sky
large_image_wcs = WCS(large_image_header)
from astropy.nddata import Cutout2D
cutout_image=Cutout2D(large_image_data,(505,506),(200,200),wcs=large_image_wcs)
#(505,506) is the centre to make the cutout and (200,200) is the size
cutout_data=cutout_image.data
cutout_wcs=cutout_image.wcs #generate the image's wcs
from astropy.visualization import (MinMaxInterval, PercentileInterval, ZScaleInterval,
                                   SqrtStretch, AsinhStretch, LogStretch, ImageNormalize)
normalization = ImageNormalize(cutout_data, interval=PercentileInterval(99.8), stretch= AsinhStretch())
fig=plt.figure(figsize=(8,6))
axis=plt.gca()
axis.imshow(cutout_data,origin='lower',cmap='nipy_spectral',  norm=normalization)
axis.xaxis.set_visible(False)
axis.yaxis.set_visible(False)
axis.text(2,190,'Cutout Image', color='white', fontsize=16)
#title of the plot
cutout_save= ('/Users/DuongHoang/Research499/Advanced_Image_Analysis/data/cutout_image.fits')
fits.writeto(cutout_save,data=cutout_data,header=cutout_wcs.to_header())
#write cutput information as fits file
plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T1_large_cutouts_of_image.png', bbox_inches='tight')
plt.show()
