import matplotlib.pyplot as plt
from astropy.io import fits
import sep
import numpy as np
large_image= fits.open('.../data/large_mosaic.fits', memmap=True)
large_image_data = large_image[0].data
large_image_header=large_image[0].header
byte_swaped_data = large_image_data.byteswap().newbyteorder() #data of the image that we want to extract the source
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

from astropy.visualization import (MinMaxInterval, PercentileInterval, ZScaleInterval,
                                   SqrtStretch, AsinhStretch, LogStretch, ImageNormalize)
object1= np.random.choice (objects,1)
object2= np.random.choice (objects,1)
object3= np.random.choice (objects,1)
object4= np.random.choice (objects,1)
object5= np.random.choice (objects,1)
from astropy.wcs import WCS #world coordinate system to map the pixels in the image to the sky
large_image_wcs = WCS(large_image_header)
#Cutout image of the slected objects from  the large image
from astropy.nddata import Cutout2D
cutout_image_1=Cutout2D(large_image_data,(object1['x'],object1['y']),(100,100),wcs=large_image_wcs)
cutout_image_2=Cutout2D(large_image_data,(object2['x'],object1['y']),(100,100),wcs=large_image_wcs)
cutout_image_3=Cutout2D(large_image_data,(object3['x'],object1['y']),(100,100),wcs=large_image_wcs)
cutout_image_4=Cutout2D(large_image_data,(object4['x'],object1['y']),(100,100),wcs=large_image_wcs)
cutout_image_5=Cutout2D(large_image_data,(object5['x'],object1['y']),(100,100),wcs=large_image_wcs)

cutout_image_data1= cutout_image_1.data
cutout_image_data2= cutout_image_2.data
cutout_image_data3= cutout_image_3.data
cutout_image_data4= cutout_image_4.data
cutout_image_data5= cutout_image_5.data
from astropy.visualization import (MinMaxInterval, PercentileInterval, ZScaleInterval,
                                   SqrtStretch, AsinhStretch, LogStretch, ImageNormalize)
normalization1 = ImageNormalize(cutout_image_data1, interval=PercentileInterval(99.8), stretch= AsinhStretch())
normalization2 = ImageNormalize(cutout_image_data2, interval=PercentileInterval(99.8), stretch= AsinhStretch())
normalization3 = ImageNormalize(cutout_image_data3, interval=PercentileInterval(99.8), stretch= AsinhStretch())
normalization4 = ImageNormalize(cutout_image_data4, interval=PercentileInterval(99.8), stretch= AsinhStretch())
normalization5 = ImageNormalize(cutout_image_data5, interval=PercentileInterval(99.8), stretch= AsinhStretch())


fig, axs = plt.subplots(nrows = 1, ncols=5,figsize=[50,8])
axis=plt.gca()
axs1,axs2,axs3,axs4,axs5 = axs

axs1.imshow(cutout_image_data1,origin='lower',cmap='gray', norm= normalization1)
axs1.xaxis.set_visible(False)
axs1.yaxis.set_visible(False)
axs1.text(2,89,'Objects 1', color='red', fontsize=14)

axs2.imshow(cutout_image_data2,origin='lower',cmap='gray',norm= normalization2)
axs2.xaxis.set_visible(False)
axs2.yaxis.set_visible(False)
axs2.text(2,89,'Objects 2', color='red', fontsize=14)

axs3.imshow(cutout_image_data3,origin='lower',cmap='gray', norm= normalization3)
axs3.xaxis.set_visible(False)
axs3.yaxis.set_visible(False)
axs3.text(2,89,'Objects 3', color='red', fontsize=14)

axs4.imshow(cutout_image_data4,origin='lower',cmap='gray', norm= normalization4)
axs4.xaxis.set_visible(False)
axs4.yaxis.set_visible(False)
axs4.text(2,89,'Objects 4', color='red', fontsize=14)

axs5.imshow(cutout_image_data5,origin='lower',cmap='gray', norm= normalization5)
axs5.xaxis.set_visible(False)
axs5.yaxis.set_visible(False)
axs5.text(2,89,'Objects 5', color='red', fontsize=14)

plt.savefig('.../plots/T1_5_slected_objects_image.png', bbox_inches='tight')
plt.show()
