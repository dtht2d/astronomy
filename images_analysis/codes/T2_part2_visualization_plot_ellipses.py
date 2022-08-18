import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sep
large_image = fits.open('/Users/DuongHoang/Research499/Advanced_Image_Analysis/codes/postage_stamp.fits',memmap=True)
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
from astropy.convolution import Gaussian2DKernel
#Define convolution filter with filter size=5
source_kernel = Gaussian2DKernel(5)
objects = sep.extract (bkg__subtracted,3, err= global_bkg.globalrms, minarea=10,
                       deblend_nthresh=64, deblend_cont=0.001, filter_kernel=source_kernel.array)
selected_objects = np.random.choice(objects,5)
print len(selected_objects)
normalization = ImageNormalize(bkg__subtracted, interval=PercentileInterval(97.), stretch= AsinhStretch())
fig=plt.figure(figsize=(8,6))
axis=plt.gca()
from matplotlib.patches import Ellipse
#plot the circle around selected_object
for i in range(len(selected_objects)):
    e = Ellipse (xy=(selected_objects['x'][i],selected_objects['y'][i]),
                 width=6*selected_objects['a'][i],
                 height=6*selected_objects['b'][i],
                 angle=selected_objects['theta'][i]*180./np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    axis.add_artist(e)
axis.imshow(bkg__subtracted, origin='lower', cmap='gray', norm=normalization)
axis.xaxis.set_visible(False)
axis.yaxis.set_visible(False)
axis.text(50,190,'Detected Object Ellipse', color='white', fontsize=18)
plt.savefig('/Users/DuongHoang/Research499/Advanced_Image_Analysis/plots/T2_part2_visualization_plot_ellipse.png',
            bbox_inches='tight')
plt.show()
